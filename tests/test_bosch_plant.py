"""P8 守門測試：機房側資料層與日前多步特徵。

每條規則兩個斷言：乾淨資料不觸發、植入壞樣本一定觸發（專案慣例：從不作響的閘門跟壞掉的閘門長得一樣）。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import bosch_plant as bp  # noqa: E402
import run_p8  # noqa: E402


def ser(pairs):
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="UTC") for t, _ in pairs])
    return pd.Series([v for _, v in pairs], index=idx, dtype=float)


# ---------------- 變動觸發 → 格點 ----------------
def test_hold_stops_at_max_hold():
    """「沒變」延續、「沒量」留白：三小時沒新紀錄，只延續 1 小時。"""
    g = bp.hold_grid(ser([("2024-01-01 00:00", 10), ("2024-01-01 03:00", 20)]), "15min", "1h")
    assert g["2024-01-01 00:00":"2024-01-01 00:45"].eq(10).all()
    assert g["2024-01-01 01:00":"2024-01-01 02:45"].isna().all()
    assert g["2024-01-01 03:00"] == 20


def test_time_weighted_not_last_value():
    """格點值是時間加權平均：0 持續 10 分鐘、30 持續 5 分鐘 → 10，不是最後一值 30。"""
    g = bp.hold_grid(ser([("2024-01-01 00:00", 0), ("2024-01-01 00:10", 30),
                          ("2024-01-01 00:15", 30)]), "15min", "1h")
    assert g["2024-01-01 00:00"] == pytest.approx(10)


def test_masked_observation_truncates_hold():
    """被遮掉的原始值（NaN）必須截斷前值延續，否則前一個值會替飽和值補位。"""
    clean = bp.hold_grid(ser([("2024-01-01 00:00", 10), ("2024-01-01 00:50", 20)]), "15min", "1h")
    assert clean["2024-01-01 00:00"] == 10 and clean["2024-01-01 00:30"] == 10
    bad = bp.hold_grid(ser([("2024-01-01 00:00", 10), ("2024-01-01 00:05", np.nan),
                            ("2024-01-01 00:50", 20)]), "15min", "1h")
    assert np.isnan(bad["2024-01-01 00:00"])      # 只剩 5 分鐘覆蓋 < 50%
    assert bad["2024-01-01 00:15":"2024-01-01 00:30"].isna().all()
    assert bad["2024-01-01 00:45"] == 20


def test_daily_meter_takes_last_before_reset():
    e = ser([("2024-06-10 06:00", 100), ("2024-06-10 23:50", 900), ("2024-06-11 00:10", 20),
             ("2024-06-11 23:50", 700), ("2024-06-12 00:10", 5), ("2024-06-12 12:00", 300)])
    tot = bp.daily_meter_totals(e)
    assert tot.tolist() == [900, 700, 300]


# ---------------- 機房側品質旗標 ----------------
@pytest.mark.parametrize("fn,clean,bad", [
    (bp.flag_saturation, [383.0, 4700.0], 10000.0),
    (bp.flag_sentinel, [-6.8, 37.4], -40.0),
    (bp.flag_trip, [5.6, 7.0, 8.1], 27.1),
])
def test_flags_clean_vs_planted(fn, clean, bad):
    assert not fn(pd.Series(clean)).any()
    assert fn(pd.Series(clean + [bad])).iloc[-1]


def test_build_plant_frame_masks_saturation_before_gridding(tmp_path):
    """端到端：植入一筆 10,000 kW 飽和值，格點上不得出現它，且被計數。"""
    t = pd.date_range("2024-06-01", periods=40, freq="2min", tz="UTC")
    vals = {"load_kw_sensor": np.full(40, 900.0), "supply_c": np.full(40, 7.0),
            "return_c": np.full(40, 12.0), "flow_m3h": np.full(40, 150.0)}
    vals["load_kw_sensor"][10] = 10_000.0
    for k, obj in bp.PLANT.items():
        if k == "setpoint_c":
            continue
        d = tmp_path / "06" / "RBHU" / obj[:4]
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"time": t, "data": vals[k]}).to_parquet(d / f"{obj}.parquet")
    obj = bp.WEATHER["outdoor_c"]
    d = tmp_path / "06" / "RBHU" / obj[:4]
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time": t, "data": np.full(40, 22.0)}).to_parquet(d / f"{obj}.parquet")
    df = bp.build_plant_frame(root=str(tmp_path))
    assert df.load_kw_sensor.max() < 1000
    assert df.n_saturation.sum() == 1
    # 自算冷量 = 150 × 5 × 1.1628 ≈ 872 kW：與感測器同量級，確認單位換算
    assert df.load_kw_physics.dropna().iloc[0] == pytest.approx(150 * 5 * bp.RHO_CP)


# ---------------- 日前多步特徵的洩漏 ----------------
def synth(days=30, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-01", periods=days * 96, freq="15min", tz="UTC")
    tod = idx.hour + idx.minute / 60
    y = pd.Series(500 + 200 * np.sin((tod - 6) / 24 * 2 * np.pi) + rng.normal(0, 10, len(idx)), index=idx)
    t = pd.Series(15 + 5 * np.sin(np.arange(len(idx)) / 96 * 2 * np.pi) + rng.normal(0, .5, len(idx)), index=idx)
    return y, t


def test_lagged_features_never_see_the_forecast_day():
    """竄改 origin 之後的所有 y 與外氣溫，lagged 特徵不得改變；perfect 特徵必須改變（探針有打到）。"""
    y, t = synth()
    rows = run_p8.build_rows(y, t)
    o = rows.origin.unique()[5]
    y2, t2 = y.copy(), t.copy()
    y2[y2.index >= o] = y2[y2.index >= o] * 7.3 + 1000
    t2[t2.index >= o] = t2[t2.index >= o] * 3.1 + 50
    rows2 = run_p8.build_rows(y2, t2)
    a = rows[rows.origin == o].reset_index(drop=True)
    b = rows2[rows2.origin == o].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[run_p8.LAGGED_COLS], b[run_p8.LAGGED_COLS])
    leaked = [c for c in run_p8.PERFECT_COLS if c not in run_p8.LAGGED_COLS and not a[c].equals(b[c])]
    assert leaked == ["t_target", "t_mean_day", "t_max_day"]


def test_origin_is_local_midnight_across_dst():
    """發預測時點是布達佩斯當地 00:00；夏令時間切換（3/31）前後都要對。"""
    y, t = synth(days=40)
    rows = run_p8.build_rows(y, t)
    local = pd.DatetimeIndex(rows.origin.unique()).tz_convert(bp.TZ)
    assert (local.hour == 0).all() and (local.minute == 0).all()
    assert pd.Timestamp("2024-04-01", tz=bp.TZ) in local
