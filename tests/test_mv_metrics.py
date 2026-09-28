"""M&V 指標守門：手算值、符號方向、以及「兩個指標各自看不到什麼」。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mv_metrics import cv_rmse, nmbe, g14_table  # noqa: E402


def test_hand_computed_values():
    y, yhat = [10, 10, 10, 10], [9, 9, 9, 9]
    assert nmbe(y, yhat) == pytest.approx(4 / (3 * 10))          # (n−p)=3
    assert cv_rmse(y, yhat) == pytest.approx(np.sqrt(4 / 3) / 10)


def test_sign_positive_means_underprediction():
    """NMBE 正＝模型低估。弄反的話，節能量的偏差方向會整個講反。"""
    y = np.full(50, 100.0)
    assert nmbe(y, y * 0.9) > 0
    assert nmbe(y, y * 1.1) < 0


def test_each_metric_is_blind_to_what_the_other_sees():
    """對稱誤差：NMBE≈0 但 CV 很大；常數偏差：兩者都有值。只看一個會漏掉另一種錯。"""
    y = np.full(100, 100.0)
    sym = y + np.tile([20.0, -20.0], 50)
    assert abs(nmbe(y, sym)) < 1e-9 and cv_rmse(y, sym) > 0.15
    assert nmbe(y, y - 5) > 0.04


def test_annual_nmbe_can_hide_offsetting_monthly_bias():
    """上半年低估 15%、下半年高估 15%：全年 NMBE≈0，逐月每一個都超過 ±5%。"""
    idx = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h", tz="UTC")
    y = pd.Series(100.0, index=idx)
    yhat = pd.Series(np.where(idx.month <= 6, 85.0, 115.0), index=idx)
    df = pd.DataFrame({"y": y, "m": yhat})
    assert abs(nmbe(df.y, df.m)) < 0.01
    monthly = df.resample("MS").mean()
    per_month = [(g.y - g.m).sum() / g.y.sum() for _, g in df.groupby(df.index.month)]
    assert all(abs(b) > 0.05 for b in per_month)
    tab = g14_table(df, "y", ["m"])
    assert abs(tab.query("granularity == 'monthly'").nmbe.iloc[0]) < 0.01   # 全期 NMBE 也被抵銷
    assert tab.query("granularity == 'monthly'").cv_rmse.iloc[0] > 0.10    # 但月度 CV 看得到
    assert len(monthly) == 12


def test_too_few_points_is_nan_not_zero():
    assert np.isnan(nmbe([1.0], [1.0])) and np.isnan(cv_rmse([], []))
