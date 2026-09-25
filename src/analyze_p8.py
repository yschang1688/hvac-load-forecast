"""P8 分析：資料層三方驗證（事前，決定目標變數用）＋三項**事後**分析（看過預測結果才想到的，只當敏感度分析）。

    python src/analyze_p8.py        # 需先跑 run_p8.py
輸出：reports/p8_analysis.txt
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bosch_plant import TZ, DAILY_METER, load_point, daily_meter_totals, physics_load_kw  # noqa: E402
from run_p8 import paired_folds  # noqa: E402

RATIO_ALERT = 1.3   # 沿用 P6 誤差監控的門檻：模型 MAE ÷ 基準 MAE 超過 1.3 倍即告警


def triangulation(tw: pd.DataFrame) -> pd.DataFrame:
    """三方投票：感測器冷量、自算冷量（流量 × ΔT）、每日能量表。以日為單位、覆蓋 ≥95% 的日才比。"""
    loads = pd.DataFrame({
        "wc003_sensor": tw.WC003_load_kw_sensor,
        "wc003_physics": physics_load_kw(tw.WC003_flow_m3h, tw.WC003_supply_c, tw.WC003_return_c),
        "wc000_sensor": tw.WC000_load_kw_sensor,
        "wc000_physics": physics_load_kw(tw.WC000_flow_m3h, tw.WC000_supply_c, tw.WC000_return_c),
    })
    loads.index = loads.index.tz_convert(TZ)
    cover = loads.notna().resample("D").mean().min(axis=1)
    daily = loads.resample("D").mean() * 24                         # kWh/日
    met = daily_meter_totals(load_point(DAILY_METER))
    met.index = pd.DatetimeIndex(met.index).tz_convert(TZ).floor("D")   # 歸零約在當地 23:57，屬當日
    daily["meter"] = met.groupby(level=0).last()
    d = daily[(cover >= 0.95)].dropna()
    d = d[d.meter > 0]
    out = d.groupby(d.index.month).apply(lambda g: pd.Series({
        "days": len(g), "meter_kwh": g.meter.mean(),
        **{f"{c}/meter": (g[c] / g.meter).median() for c in loads.columns}}))
    return out


def regime_shift(daily: pd.DataFrame, tw: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """10 月跳脫後的工況切換：基載、模型 vs 昨天基準的誤差比、P6 門檻何時響。"""
    y = tw.WC003_load_kw_sensor.copy()
    y.index = y.index.tz_convert(TZ)
    night = y.between_time("00:00", "05:00").resample("D").mean()
    w = daily.pivot_table(index="local_date", columns="method", values="mae")
    w.index = pd.to_datetime(w.index)
    roll = w.rolling(7, min_periods=5).mean()
    ratio = (roll.lgbm_lagged / roll.sn_day).rename("ratio_7d")
    after = ratio["2024-10-16":]
    first = after[after > RATIO_ALERT].index.min()
    back = after[(after.index > first) & (after < RATIO_ALERT)].index.min() if pd.notna(first) else None
    tab = pd.DataFrame({"night_base_kw": night.tz_localize(None)["2024-10-06":"2024-11-15"],
                        "ratio_7d": ratio["2024-10-06":"2024-11-15"]}).round(2)
    txt = (f"跳脫後首次誤差比 > {RATIO_ALERT}：{first.date() if pd.notna(first) else '未觸發'}；"
           f"回落到門檻下：{back.date() if back is not None and pd.notna(back) else '期間內未回落'}")
    return txt, tab


def weather_after_summer(daily: pd.DataFrame) -> str:
    """完美預報 vs 只用滯後，只看已經見過一次夏天之後的 fold（8–12 月）。事後切片。"""
    sub = daily.fold >= 8
    parts = []
    for metric in ["mae", "day_ape"]:
        r = paired_folds(daily, "lgbm_perfect", "lgbm_lagged", metric, subset=sub)
        parts.append(f"  {metric}: perfect {r['mean_a']:.4g} vs lagged {r['mean_b']:.4g}  "
                     f"p={r['p']:.3g}  較佳 fold {r['folds_a_better']}  判定={r['verdict']}")
    return "\n".join(parts)


def twin_correlation(tw: pd.DataFrame) -> pd.DataFrame:
    """共線洩漏：官方目標 WC000.AM02（回水溫）與同水路另兩個量測點的同時刻相關。"""
    c = tw[["WC000_return_c", "WC001_return_c", "WC003_return_c"]].dropna()
    return c.corr().round(4)


def main():
    tw = pd.read_parquet(ROOT / "data/processed/bosch_twins_15min.parquet")
    daily = pd.read_csv(ROOT / "reports/p8_daily.csv")
    lines = ["# P8 分析輸出", "", "## 一、三方驗證（事前；逐月中位比值，對每日能量表）"]
    lines.append(triangulation(tw).round(3).to_string())
    txt, tab = regime_shift(daily, tw)
    lines += ["", "## 二、10 月跳脫後工況切換（事後）", txt, tab.to_string()]
    lines += ["", "## 三、完美預報 vs 滯後，只看 8–12 月（事後切片）", weather_after_summer(daily)]
    lines += ["", "## 四、官方目標的共線分身（同時刻相關）", twin_correlation(tw).to_string()]
    out = "\n".join(lines)
    (ROOT / "reports/p8_analysis.txt").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
