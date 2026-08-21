"""P3 守門測試：架構層的洩漏與基準線的正確性。

P2 的行為檢測只看特徵，看不到注意力矩陣——**架構層可以獨立製造洩漏**，
所以 causal mask 需要自己的釘子。
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import models  # noqa: E402  （必須早於 torch：models 內部先載 lightgbm，見該檔頂端註解）
import torch  # noqa: E402


def test_transformer_causal_mask_blocks_future():
    """改動未來第 k 步的輸入，第 1…k-1 步的輸出必須逐位元不變。

    反向斷言同樣重要：第 k 步本身要變（否則可能是模型根本沒吃 future），
    改動過去也要有影響（否則可能是輸出與輸入無關）。
    """
    torch.manual_seed(0)
    m = models.TimeSeriesTransformer(6, 5, 24).eval()
    p, f = torch.randn(4, 168, 6), torch.randn(4, 24, 5)
    with torch.no_grad():
        a = m(p, f).numpy()
        for k in (5, 12, 23):
            f2 = f.clone(); f2[:, k, :] += 99.0
            b = m(p, f2).numpy()
            assert np.abs(a[:, :k] - b[:, :k]).max() == 0.0, f"第 {k} 步的輸入影響到了更早的輸出"
            assert np.abs(a[:, k] - b[:, k]).max() > 0, f"第 {k} 步的輸入完全沒被用到"
        p2 = p.clone(); p2[:, 0, :] += 99.0
        assert np.abs(a - m(p2, f).numpy()).max() > 0, "過去序列對輸出沒有影響"


def test_seasonal_naive_uses_last_week():
    idx = pd.date_range("2016-01-01", periods=24 * 40, freq="h")
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    m = models.SeasonalNaive(period=168)
    origins = idx[300:305]
    out = m.predict_from_series(y, origins, 24)
    for i, t in enumerate(origins):
        p = idx.get_loc(t)
        for h in range(1, 25):
            assert out[i, h - 1] == y.iloc[p + h - 168]


def test_scaler_fitted_on_train_only():
    """標準化只能用訓練段統計量——用到測試段就是分布資訊洩漏。"""
    tr = np.random.RandomState(0).normal(0, 1, (100, 10, 3)).astype(np.float32)
    te = tr * 50 + 100
    s = models._Scaler().fit(tr)
    mu_before = s.mu.copy()
    _ = s(te)
    assert np.allclose(s.mu, mu_before), "呼叫 transform 後統計量被改動了"
    assert np.abs(s(te)).mean() > 5, "測試段本該偏離訓練分布，若已被歸一代表用了測試統計量"


def test_make_sequences_never_reaches_beyond_horizon():
    """序列組裝：past 只到 t，future 與標的只到 t+H，不得多取一格。"""
    idx = pd.date_range("2016-01-01", periods=24 * 60, freq="h")
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    wx = pd.DataFrame({"airTemperature": np.arange(len(idx), dtype=float),
                       "dewTemperature": np.zeros(len(idx))}, index=idx)
    L, H = 168, 24
    got = models.make_sequences(y, wx, idx[400:403], L, H)
    past, fut, tgt, keep = got
    for i, t in enumerate(keep):
        p = idx.get_loc(t)
        assert past[i, -1, 0] == y.iloc[p], "past 最後一格必須正好是 t"
        assert tgt[i, 0] == y.iloc[p + 1] and tgt[i, -1] == y.iloc[p + H]
        assert fut[i, 0, 0] == wx["airTemperature"].iloc[p + 1]
        assert fut[i, -1, 0] == wx["airTemperature"].iloc[p + H]


def test_sequences_reject_nan_targets():
    """標的含 NaN 的序列必須在組裝階段就被剔除。

    這條擋的是一個**靜默**失敗：NaN 標的 → loss 變 nan → `nan < best` 恆為 False →
    early stopping 永不更新、val loss 停在 inf → 上層把整個架構跳過，
    而全程沒有任何例外或警告。實跑時的症狀是「結果表裡就是沒有神經網路那幾列」。
    """
    idx = pd.date_range("2016-01-01", periods=24 * 60, freq="h")
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    y.iloc[500:520] = np.nan                       # 標的區間打洞
    wx = pd.DataFrame({"airTemperature": np.ones(len(idx)),
                       "dewTemperature": np.ones(len(idx))}, index=idx)
    got = models.make_sequences(y, wx, idx[400:500], 120, 24)
    assert got is not None
    past, fut, tgt, keep = got
    assert not np.isnan(tgt).any(), "標的仍含 NaN"
    assert not np.isnan(past).any() and not np.isnan(fut).any()


def test_train_nn_raises_on_non_finite_loss():
    """驗證損失非有限值必須明確報錯，不得靜默回 inf。"""
    n, L, H = 64, 24, 6
    tr = (np.random.rand(n, L, 3).astype(np.float32),
          np.random.rand(n, H, 2).astype(np.float32),
          np.random.rand(n, H).astype(np.float32))
    bad = list(tr)
    bad[2] = bad[2].copy(); bad[2][0, 0] = np.nan
    net = models.LSTMSeq2Seq(3, 2, H, hidden=8)
    with pytest.raises(ValueError):
        models.train_nn(net, tr, tuple(bad), epochs=1, patience=1)
