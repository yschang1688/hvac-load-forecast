"""P7 守門測試：關機近似規則。每一條都對應一種會讓 P3 聚合靜默出錯的情況。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import operating_state as OS  # noqa: E402


def _office_week(start="2017-03-06"):
    """辦公建築的正常一週：平日 8–18 時有負荷，夜間與週末為 0。"""
    idx = pd.date_range(start, periods=168, freq="h")
    on = (idx.dayofweek < 5) & (idx.hour >= 8) & (idx.hour < 18)
    return pd.Series(np.where(on, 100.0, 0.0), index=idx)


def test_normal_nights_and_weekends_are_not_shutdown():
    """正常辦公週有 70% 的時數是 0（夜間＋週末），但這不是停機。
    若規則用零值佔比，這一週會被誤判——正是本模組不用零值佔比的原因。"""
    y = _office_week()
    assert (y == 0).mean() > 0.6
    state, share = OS.window_state(y)
    assert state == "operating", share


def test_weekend_run_below_threshold_is_not_counted():
    """週五 18 時到週一 8 時是 62 小時的零值段——長於 24 小時。
    所以辦公週的 share 不是 0，但必須低於 50%，否則每個辦公週都會被排除。"""
    _, share = OS.window_state(_office_week())
    assert 0 < share < 0.5, share


def test_seasonal_shutdown_is_detected():
    idx = pd.date_range("2017-01-02", periods=168, freq="h")
    y = pd.Series(0.0, index=idx)
    y.iloc[50:54] = 5.0                          # 冬季偶發短暫開機（P3 關機視窗的實況）
    state, share = OS.window_state(y)
    assert state == "shutdown" and share > 0.9


def test_missing_values_break_zero_runs():
    """斷訊不能被算成停機：NaN 中斷零值段。"""
    idx = pd.date_range("2017-01-02", periods=48, freq="h")
    y = pd.Series(0.0, index=idx)
    y.iloc[::10] = np.nan                        # 每 10 小時一個缺值 → 最長零值段 9 小時
    assert not OS.zero_run_mask(y, 24).any()


def test_mostly_missing_window_is_unknown():
    idx = pd.date_range("2017-01-02", periods=168, freq="h")
    y = pd.Series(np.nan, index=idx)
    y.iloc[:40] = 0.0
    assert OS.window_state(y)[0] == "unknown"


def test_context_prevents_truncating_runs_at_window_edge():
    """一段從上週延續過來的停機，在本週只剩 20 小時。沒有 context 會漏判。"""
    idx = pd.date_range("2017-01-01", periods=24 * 14, freq="h")
    y = pd.Series(100.0, index=idx)
    y.iloc[24 * 7 - 30: 24 * 7 + 20] = 0.0       # 跨越第 7 天邊界的 50 小時停機
    week2 = y.iloc[24 * 7:]
    assert OS.zero_run_mask(week2, 24).sum() == 0                     # 單看本週：20 小時，未達門檻
    assert OS.shutdown_share(week2, 24, context=y) > 0                # 有 context：正確計入
