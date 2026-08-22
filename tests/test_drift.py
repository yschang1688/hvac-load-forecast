"""P4 守門測試：PSI 的三個實作陷阱各有一根釘子。"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import drift  # noqa: E402


def test_same_distribution_is_near_zero():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 8000)
    e = drift.psi_bins(ref)
    assert drift.psi(ref, rng.normal(0, 1, 2000), e) < 0.05


def test_shifted_distribution_is_detected():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 8000)
    e = drift.psi_bins(ref)
    assert drift.psi(ref, rng.normal(1.0, 1, 2000), e) > 0.25


def test_empty_bins_from_sparse_sampling_do_not_fake_drift():
    """**同一個分布**只是抽樣稀疏而產生空箱時，PSI 不得因此判成漂移。

    初版把比例 clip 到 1e-6，每個空箱固定貢獻約 1.1，與樣本數無關；
    稀疏視窗於是被判成大幅漂移。改 Laplace 平滑後，空箱機率隨樣本數縮放。
    ⚠ 注意這條測的是**假漂移**：真的高度集中的視窗本來就該得到高 PSI，那不是 bug。
    """
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 8000)
    e = drift.psi_bins(ref)
    worst = max(drift.psi(ref, rng.normal(0, 1, 40), e) for _ in range(30))  # 40 點必有空箱
    assert worst < 1.0, f"同分布的稀疏視窗最大 PSI={worst:.2f}，空箱仍在製造假漂移"


def test_small_window_same_distribution_still_stable():
    """樣本數小不該本身就觸發告警——否則監控週期一縮短，告警就全滿。"""
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 8000)
    e = drift.psi_bins(ref)
    vals = [drift.psi(ref, rng.normal(0, 1, 168), e) for _ in range(20)]
    assert max(vals) < 0.25, f"同分布的 168 點視窗最大 PSI={max(vals):.3f}，已達 alert"


def test_bins_come_from_reference_and_are_frozen():
    """分箱邊界必須來自參考期。用當期資料重新分箱，兩邊的分布定義都變了。"""
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, 5000)
    e1 = drift.psi_bins(ref)
    e2 = drift.psi_bins(ref)
    assert np.allclose(e1, e2)
    assert e1[0] == -np.inf and e1[-1] == np.inf


def test_degenerate_series_does_not_pretend_to_judge():
    """常數序列（如整段關機）無法建立有意義的分箱，應退化成不判而非給假數字。"""
    e = drift.psi_bins(np.zeros(1000))
    assert len(e) == 2


def test_verdict_uses_max_not_mean():
    """以最大單欄 PSI 判定——平均會被大量穩定欄位稀釋掉一個真的壞掉的特徵。"""
    state, m = drift.verdict({"a": 0.01, "b": 0.02, "c": 0.9})
    assert state == "alert" and m == 0.9


def test_concept_drift_injection_changes_mapping():
    import pandas as pd
    idx = pd.date_range("2017-01-01", periods=24 * 30, freq="h")
    y = pd.Series(np.sin(np.arange(len(idx)) / 12) * 10 + 50, index=idx)
    out = drift.inject_concept_drift(y, idx[24 * 15], scale=1.6, shift_hours=4)
    assert np.allclose(out[: 24 * 15], y[: 24 * 15]), "注入點之前不該被改動"
    assert not np.allclose(out[24 * 15:], y[24 * 15:]), "注入點之後應該改變"
