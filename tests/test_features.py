"""P2 守門測試：時序洩漏的兩種型態各有對應的釘子。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features  # noqa: E402


def synth(n=24 * 400, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n, freq="h")
    h = idx.hour.to_numpy()
    y = pd.Series(np.clip(60 + 40 * np.sin((h - 6) / 24 * 2 * np.pi) + rng.normal(0, 3, n), 1, None), index=idx)
    wx = pd.DataFrame({
        "airTemperature": 25 + 8 * np.sin(np.arange(n) / 24) + rng.normal(0, 1, n),
        "dewTemperature": 15 + rng.normal(0, 1, n),
        "cloudCoverage": rng.uniform(0, 8, n),
        "windSpeed": rng.uniform(0, 10, n),
    }, index=idx)
    return y, wx


@pytest.mark.parametrize("mode", ["perfect_forecast", "lagged_only"])
def test_base_design_never_looks_ahead(mode):
    """紅線：build_design 的任何欄位都不得依賴 t 之後的資料，兩種 mode 皆然。"""
    y, wx = synth()
    assert features.detect_lookahead(y, wx, {"weather_mode": mode}) == []


def test_leak_probe_is_caught():
    """探針必須真的壞：植入一個偷看未來 24h 的特徵，偵測器一定要點名它。

    沒有這條，上面那條「偵測器沒抓到東西」就可能只是因為偵測器本身壞了。
    """
    y, wx = synth()
    bad = features.detect_lookahead(y, wx, {"weather_mode": "lagged_only", "_inject_leak": True})
    assert "LEAK_future_mean24" in bad


def test_future_weather_is_opt_in_and_declared():
    """未來天氣只能來自 future_weather_block，且必須由 weather_mode 明碼開啟。"""
    y, wx = synth()
    idx = y.index
    on = features.future_weather_block(wx, idx, 24, {"weather_mode": "perfect_forecast"})
    off = features.future_weather_block(wx, idx, 24, {"weather_mode": "lagged_only"})
    assert on.shape[1] > 0 and off.shape[1] == 0
    assert all(c.startswith("fut_") for c in on.columns), "未來欄位必須以 fut_ 前綴自我標示"
    # 該區塊確實取到 t+h 的值
    assert np.isclose(on["fut_airTemperature_h1"].iloc[10], wx["airTemperature"].iloc[11])
    assert np.isclose(on["fut_airTemperature_h24"].iloc[10], wx["airTemperature"].iloc[34])


def test_targets_are_future_shifted():
    y, _ = synth(n=500)
    t = features.build_targets(y, 24)
    assert list(t.columns) == [f"y_h{h}" for h in range(1, 25)]
    assert np.isclose(t["y_h1"].iloc[0], y.iloc[1])
    assert np.isclose(t["y_h24"].iloc[0], y.iloc[24])
    assert t["y_h24"].iloc[-24:].isna().all(), "序列尾端沒有未來，標的必須是 NaN"


def test_rolling_stats_exclude_current_row():
    """滾動統計必須自 shift(1) 起算——含當下那格就是用答案算特徵。"""
    y, wx = synth(n=1000)
    X = features.build_design(y, wx, {"weather_mode": "lagged_only"})
    manual = y.shift(1).rolling(24, min_periods=6).mean()
    pd.testing.assert_series_equal(X["load_roll_mean_24"], manual, check_names=False)


def test_no_dropna_inside_builder():
    """設計矩陣不得在建構階段丟列：樣本存在與否若取決於未來，切分就已經被污染。"""
    y, wx = synth(n=1000)
    X = features.build_design(y, wx, {"weather_mode": "lagged_only"})
    assert len(X) == len(y)
    assert X.iloc[:168].isna().any().any(), "前段本該因 lag 而有 NaN，若無代表被填補或丟棄了"


def test_rolling_origin_never_touches_sealed_test():
    """切分守衛：任何模型選擇用的 fold 都不得伸進封存測試期。

    探針：把封存起點往前推到第一個 fold 的中間，切分函式必須翻臉。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import dataset
    cfg = dataset.load_cfg()
    idx = pd.date_range("2016-01-01", "2017-12-31 23:00", freq="h")
    splits = dataset.rolling_origin_splits(idx, cfg)
    sealed = pd.Timestamp(cfg["modeling"]["sealed_test_start"])
    assert all(s.valid_end <= sealed for s in splits)

    bad = {**cfg, "modeling": {**cfg["modeling"], "sealed_test_start": "2017-02-01"}}
    with pytest.raises(AssertionError):
        dataset.rolling_origin_splits(idx, bad)


def test_train_valid_gap_equals_horizon():
    """訓練段尾端的標的伸到 t+H，所以驗證段必須至少晚 H 小時開始。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import dataset
    cfg = dataset.load_cfg()
    idx = pd.date_range("2016-01-01", "2017-12-31 23:00", freq="h")
    H = cfg["modeling"]["horizon"]
    for s in dataset.rolling_origin_splits(idx, cfg):
        assert (s.valid_start - s.train_end) >= pd.Timedelta(hours=H)
