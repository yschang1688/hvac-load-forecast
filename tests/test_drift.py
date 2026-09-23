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


# ---- P6：誤差監控 -----------------------------------------------------------

def test_error_monitor_stable_when_error_stays_at_baseline():
    """誤差沒惡化就該說 stable——PSI 在本場域做不到這件事（624/624 alert）。"""
    rng = np.random.default_rng(4)
    base = drift.error_baseline(rng.normal(0.6, 0.05, 8))
    states = [drift.error_verdict(v, base)[0] for v in rng.normal(0.6, 0.05, 30)]
    assert states.count("alert") <= 2, states


def test_error_monitor_alerts_when_error_degrades():
    base = drift.error_baseline([0.6, 0.62, 0.58, 0.61, 0.59])
    state, ratio = drift.error_verdict(0.9, base)
    assert state == "alert" and ratio > 1.3


def test_error_baseline_uses_median_not_mean():
    """基準期若含一個關機週（skill=7），平均會被拖到 1.9，中位數仍在 0.6 附近。
    用平均的話，之後真正的惡化（0.9）會被判成 stable。"""
    base = drift.error_baseline([0.6, 0.62, 7.0, 0.61, 0.59])
    assert base < 0.7
    assert drift.error_verdict(0.9, base)[0] == "alert"


def test_error_monitor_does_not_pretend_to_judge_without_baseline():
    """基準期樣本不足或當週無誤差（整週關機、無實際值）→ unknown，不得給假判定。"""
    assert np.isnan(drift.error_baseline([0.6, 0.7]))
    assert drift.error_verdict(0.9, np.nan)[0] == "unknown"
    assert drift.error_verdict(np.nan, 0.6)[0] == "unknown"
    assert drift.error_verdict(0.9, 0.0)[0] == "unknown"


def test_error_monitor_catches_concept_drift_that_psi_misses():
    """概念漂移：輸入分布不動、映射改變。PSI 對輸入算出來仍是 stable，
    誤差監控看凍結模型的誤差則必須 alert。這是 P4 §三的結構性盲區，釘成測試。"""
    import pandas as pd
    rng = np.random.default_rng(5)
    idx = pd.date_range("2017-01-01", periods=24 * 60, freq="h")
    x = pd.Series(np.sin(np.arange(len(idx)) / 12) * 10 + 50, index=idx)      # 輸入（如外氣）
    y_true = x * 2.0 + rng.normal(0, 1.0, len(idx))                            # 原映射＋量測雜訊
    y_drift = drift.inject_concept_drift(y_true, idx[24 * 30], scale=1.6, shift_hours=4)
    ref = x[: 24 * 30].to_numpy()
    edges = drift.psi_bins(ref)
    psi_post = drift.psi(ref, x[24 * 30:].to_numpy(), edges)
    assert psi_post < 0.10, f"輸入分布未變，PSI 應 stable，實得 {psi_post:.3f}"

    pred = x * 2.0                                                              # 凍結模型（學到原映射）
    weeks_pre = [float((pred - y_drift)[i:i + 168].abs().mean()) for i in range(0, 24 * 30, 168)]
    base = drift.error_baseline(weeks_pre)
    assert 0.5 < base < 1.5, f"注入前基準誤差應在雜訊尺度（≈0.8），實得 {base:.3f}"
    err_post = float((pred - y_drift)[24 * 30:].abs().mean())
    assert drift.error_verdict(err_post, base)[0] == "alert", (base, err_post)


def test_should_promote_keeps_old_model_when_retrain_did_not_improve():
    """重訓不是信仰：驗收沒變好就不換版本。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import schedule
    assert schedule.should_promote(0.60, 0.55)[0] is True
    assert schedule.should_promote(0.60, 0.66)[0] is False
    assert schedule.should_promote(None, 0.66)[0] is True          # 首版放行
    assert schedule.should_promote(0.60, None)[0] is False         # 無驗收不得晉升
