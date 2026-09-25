"""P8｜機房側資料層：Bosch Budapest 冰水機房 BMS（ELIAS UC1，Zenodo 2024 全年，CC BY 4.0）。

BDG2（P1–P7）是**建築側**的冰水表讀數；這份是**機房側**：供回水溫、流量、需求冷量、
冰水機狀態、泵轉速、設定點。資料取得與授權見 `docs/P8_PLAN.md`。

本檔處理的是 BDG2 沒有的一個問題：**變動觸發取樣**（change-triggered）。
BMS 只在數值變動超過死區時寫一筆，所以兩筆之間沒有紀錄有兩種可能：
  (a) 值沒變——該用前值延續（sample-and-hold）
  (b) 沒在量——斷線、當機、記錄器停擺
官方範例用 10 分鐘 ffill，會把 (b) 也當成 (a)。本檔的做法是**延續但設停滯上限**：
距上一筆超過 `max_hold` 就判定為「沒量」，該段不進格點平均。
上限依點位自身的取樣節奏設（冰水迴路 Δt p99 約 0.1–0.3 h，外氣溫約 0.8 h），不是一個全域常數。
"""
from __future__ import annotations
import glob
import os
import numpy as np
import pandas as pd

ROOT = os.environ.get("BOSCH_INTERIM", "data/interim/bosch")
TZ = "Europe/Budapest"

# 水的體積比熱：4.186 kJ/(kg·K) × 1000 kg/m³ ÷ 3600 s/h = 1.1628 kW/(m³/h·K)
RHO_CP = 4.186 * 1000 / 3600

# 冰水主迴路（BP201/202/206）。官方範例的目標變數是 WC000.AM02（回水溫）。
PLANT = {
    "supply_c": "B205WC000.AM01",
    "return_c": "B205WC000.AM02",
    "flow_m3h": "B205WC000.AM71",
    "load_kw_sensor": "B205WC000.AM71_2",
    "setpoint_c": "B205WC000.VT01",
}
# 同一條水路的其他量測點（metadata 標示同為 BP201/202/206）——共線洩漏候選
TWINS = {
    "WC001": {"supply_c": "B205WC001.AM01", "return_c": "B205WC001.AM02",
              "flow_m3h": "B205WC001.AM71", "load_kw_sensor": "B205WC001.AM71_2"},
    "WC003": {"supply_c": "B205WC003.AM01", "return_c": "B205WC003.AM02",
              "flow_m3h": "B205WC003.AM71", "load_kw_sensor": "B205WC003.AM71_2"},
}
WEATHER = {"outdoor_c": "B106WS01.AM54", "rh_pct": "B106WS01.AM53"}
DAILY_METER = "B205WC003.PA71_2_D"          # 當日累計冷量 kWh，每日歸零
CHILLER_POWER = ["B205WC010.AM55_2", "B205WC030.AM55_2"]

# 品質門檻（物理理由見各函式 docstring）
SENSOR_SATURATION_KW = 9_999.0
OUTDOOR_SENTINEL_C = -30.0
TRIP_SUPPLY_C = 15.0


def load_point(obj: str, root: str = ROOT) -> pd.Series:
    """把 12 個月的單一點位接成一條序列（UTC，依時間排序、去重）。"""
    fs = sorted(glob.glob(f"{root}/*/RBHU/{obj[:4]}/{obj}.parquet"))
    if not fs:
        raise FileNotFoundError(obj)
    d = pd.concat([pd.read_parquet(f) for f in fs])
    d = d.drop_duplicates("time").sort_values("time")
    return d.set_index("time")["data"].rename(obj)


def hold_grid(s: pd.Series, freq: str = "15min", max_hold: str = "1h",
              min_cover: float = 0.5) -> pd.Series:
    """變動觸發序列 → 等距格點的**時間加權平均**，延續前值但不超過 `max_hold`。

    每一筆觀測的值延續到下一筆或 `max_hold` 為止（取先到者）；超過的部分視為沒量。
    **值為 NaN 的觀測（上游品質旗標遮掉的原始值）也會截斷前一筆的延續**，
    否則遮掉一個飽和值，前一個值就會被延續去填它的位置。
    格點內有效覆蓋低於 `min_cover` 則回 NaN，而不是用少數幾分鐘代表整格。
    """
    if s.dropna().empty:
        return pd.Series(dtype=float)
    step = pd.Timedelta(freq)
    stp = step.value
    t = s.index
    grid = pd.date_range(t[0].floor(freq), t[-1].ceil(freq), freq=freq)
    a = t.asi8
    t_next = np.append(a[1:], a[-1] + stp)
    b = np.minimum(t_next, a + pd.Timedelta(max_hold).value)
    v = s.to_numpy(dtype=float)
    g0 = grid.asi8[0]
    num = np.zeros(len(grid))
    den = np.zeros(len(grid))
    i0 = (a - g0) // stp
    i1 = (b - 1 - g0) // stp
    for k in np.flatnonzero(~np.isnan(v)):
        for gi in range(int(i0[k]), int(i1[k]) + 1):
            if 0 <= gi < len(grid):
                cs = max(a[k], g0 + gi * stp)
                ce = min(b[k], g0 + (gi + 1) * stp)
                if ce > cs:
                    num[gi] += v[k] * (ce - cs)
                    den[gi] += ce - cs
    out = np.where(den >= min_cover * stp, num / np.where(den > 0, den, 1), np.nan)
    return pd.Series(out, index=grid, name=s.name)


def daily_meter_totals(e: pd.Series) -> pd.Series:
    """每日累計計數器 → 各日總量。取「歸零前最後一值」，不取 max（跨日漏記時 max 會混到隔日）。"""
    e = e.dropna()
    reset = e.diff() < 0
    day_id = reset.cumsum()
    last = e.groupby(day_id).agg(["last", "size"])
    stamp = e.index.to_series().groupby(day_id).last()
    out = pd.Series(last["last"].to_numpy(), index=stamp.to_numpy())
    return out


def physics_load_kw(flow_m3h: pd.Series, supply_c: pd.Series, return_c: pd.Series) -> pd.Series:
    """Q = 流量 × ΔT × ρc。與 BMS 的「需求冷量」點位互為獨立來源（自算 vs 感測器）。"""
    dT = (return_c - supply_c)
    return (flow_m3h * dT * RHO_CP).rename("load_kw_physics")


# ---------------- 機房側品質旗標（BDG2 沒有的四型） ----------------
def flag_saturation(load_kw: pd.Series) -> pd.Series:
    """需求冷量點位的 10,000 kW 是量程上限（溢位），不是真實負荷——實際峰值約 4,700 kW。"""
    return load_kw >= SENSOR_SATURATION_KW


def flag_sentinel(outdoor_c: pd.Series) -> pd.Series:
    """外氣溫 −40 °C 是氣象站的故障哨兵值；布達佩斯 7 月不可能出現。"""
    return outdoor_c <= OUTDOOR_SENTINEL_C


def flag_trip(supply_c: pd.Series) -> pd.Series:
    """冰水供水溫 > 15 °C（設定點 7 °C）＝冰水機跳脫或停機。這段的『負荷』不是需求，是事故。"""
    return supply_c > TRIP_SUPPLY_C


def build_plant_frame(freq: str = "15min", root: str = ROOT) -> pd.DataFrame:
    """主迴路 + 外氣的格點資料框（UTC 索引）。

    原始檔不改；違規原始值在**格點化之前**遮成 NaN（否則飽和值會被平均進格點），
    遮了幾筆記在旗標欄，事後查得到。
    """
    raw = {k: load_point(obj, root) for k, obj in PLANT.items() if k != "setpoint_c"}
    raw["outdoor_c"] = load_point(WEATHER["outdoor_c"], root)
    sat = flag_saturation(raw["load_kw_sensor"])
    sen = flag_sentinel(raw["outdoor_c"])
    raw["load_kw_sensor"] = raw["load_kw_sensor"].mask(sat)
    raw["outdoor_c"] = raw["outdoor_c"].mask(sen)
    df = pd.DataFrame({k: hold_grid(v, freq, "2h" if k == "outdoor_c" else "1h")
                       for k, v in raw.items()})
    df["n_saturation"] = sat[sat].resample(freq).size().reindex(df.index, fill_value=0)
    df["n_sentinel"] = sen[sen].resample(freq).size().reindex(df.index, fill_value=0)
    df["flag_trip"] = flag_trip(df["supply_c"]).fillna(False).astype(bool)
    df["load_kw_physics"] = physics_load_kw(df["flow_m3h"], df["supply_c"], df["return_c"])
    return df


def build_twins_frame(freq: str = "15min", root: str = ROOT) -> pd.DataFrame:
    """主迴路與兩個分身量測點（WC001、WC003）的格點資料框，欄名 `<迴路>_<量>`。三方投票與預測目標用。"""
    pts = {"WC000": {k: v for k, v in PLANT.items() if k != "setpoint_c"}, **TWINS}
    cols = {}
    for loop, d in pts.items():
        for k, obj in d.items():
            s = load_point(obj, root)
            if k == "load_kw_sensor":
                s = s.mask(flag_saturation(s))
            cols[f"{loop}_{k}"] = hold_grid(s, freq, "1h")
    return pd.DataFrame(cols)


if __name__ == "__main__":
    out = os.path.join("data", "processed")
    os.makedirs(out, exist_ok=True)
    build_plant_frame().to_parquet(os.path.join(out, "bosch_plant_15min.parquet"))
    build_twins_frame().to_parquet(os.path.join(out, "bosch_twins_15min.parquet"))
    print("wrote", out)
