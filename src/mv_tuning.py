"""P12｜簡化版節能基準的調校候選：月份分段、佔用分群、正則。

設計在 docs/P12_PLAN.md 事前登記。每個候選仍然**只吃獨立變數**，基準期訓練後凍結；
`occ`（高／低負載時段）只由**訓練資料**的能耗決定，預測對象那段的能耗不得流進來。
C0＝Spec("none", "shared", 1e-3) 與 `mv_baseline.frozen_towt` 等價。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

import mv_baseline as mb

SEGS = ("none", "m3")
TEMPS = ("shared", "occ")
RIDGES = (1e-3, 1.0, 10.0)
SIMPLICITY = [("none", "shared"), ("none", "occ"), ("m3", "shared"), ("m3", "occ")]   # 由簡到繁


@dataclass(frozen=True)
class Spec:
    seg: str = "none"
    temp: str = "shared"
    ridge: float = 1e-3

    @property
    def name(self) -> str:
        return f"{self.seg}/{self.temp}/ridge={self.ridge:g}"


ALL_SPECS = [Spec(s, t, r) for s, t in SIMPLICITY for r in RIDGES]
C0 = Spec()


def _tow(idx: pd.DatetimeIndex) -> np.ndarray:
    return (idx.dayofweek * 24 + idx.hour).to_numpy()


def occupied_tow(train: pd.DataFrame) -> np.ndarray:
    """168 個時段裡，平均冷量高於「各時段平均的中位數」者算高負載。只看訓練資料。"""
    means = np.full(168, np.nan)
    g = train.kw.groupby(_tow(train.index)).mean()
    means[g.index.to_numpy()] = g.to_numpy()
    return means > np.nanmedian(means)


def design(frame: pd.DataFrame, occ: np.ndarray | None) -> np.ndarray:
    """時段變數＋外氣溫分段線性（shared 一組；occ 高低負載各一組）＋假日。沒有任何能耗欄位。"""
    tow = _tow(frame.index)
    t = frame.t.to_numpy(dtype=float)
    T = np.column_stack([t] + [np.maximum(t - k, 0.0) for k in mb.KNOTS])
    if occ is None:
        temp = T
    else:
        o = occ[tow].astype(float)[:, None]
        temp = np.hstack([T * o, T * (1.0 - o)])
    return np.hstack([np.eye(168)[tow], temp, frame[["holiday"]].to_numpy(dtype=float)])


def month_weights(months: np.ndarray, m: int) -> np.ndarray:
    """three_month_weighted：當月 1、前後月 0.5（12 月與 1 月相鄰）、其餘 0。"""
    d = np.abs(months - m)
    d = np.minimum(d, 12 - d)
    return np.where(d == 0, 1.0, np.where(d == 1, 0.5, 0.0))


def _penalty(n_cols: int) -> np.ndarray:
    """正則矩陣：時段變數只懲罰「偏離 168 個時段平均」的部分，不懲罰水位；其餘係數照常懲罰。

    模型沒有截距，水位由 168 個時段變數共同承擔。若對它們直接做 ridge，係數被往 0 壓，
    調整後基準就整體偏低——等於憑空吃掉節能（合成資料實測：ridge=1 時植入 15% 只算回 12.6%）。
    """
    P = np.eye(n_cols)
    P[:168, :168] -= 1.0 / 168.0
    return P


def _solve(X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float) -> np.ndarray:
    Xw = X * w[:, None]
    return np.linalg.solve(X.T @ Xw + ridge * _penalty(X.shape[1]), Xw.T @ y)


def predict(spec: Spec, train: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """用 train 擬合、對 target 的獨立變數算調整後基準。target 的 kw 欄不被讀取。"""
    train = train.dropna(subset=["kw", "t"])
    occ = occupied_tow(train) if spec.temp == "occ" else None
    Xtr, ytr = design(train, occ), train.kw.to_numpy(dtype=float)
    tgt = target.dropna(subset=["t"])
    Xtg = design(tgt, occ)
    out = np.full(len(tgt), np.nan)
    if spec.seg == "none":
        out[:] = Xtg @ _solve(Xtr, ytr, np.ones(len(train)), spec.ridge)
    else:
        tr_m, tg_m = train.index.month.to_numpy(), tgt.index.month.to_numpy()
        for m in np.unique(tg_m):
            w = month_weights(tr_m, m)
            keep = w > 0
            if not keep.any():
                continue                                  # 該月份沒有任何訓練資料：留空，不外推
            out[tg_m == m] = Xtg[tg_m == m] @ _solve(Xtr[keep], ytr[keep], w[keep], spec.ridge)
    return pd.Series(out, index=tgt.index, name="adj_baseline").reindex(target.index)


def builder(spec: Spec) -> mb.Builder:
    def _b(frame: pd.DataFrame, report_start: pd.Timestamp) -> pd.Series:
        rep = frame[frame.index >= report_start]
        return predict(spec, frame[frame.index < report_start], rep)
    return _b


def week_folds(idx: pd.DatetimeIndex, k: int = 4) -> np.ndarray:
    """整週交錯：ISO 週數除以 k 的餘數。每一折的留出週散在全年。"""
    return (idx.isocalendar().week.to_numpy().astype(int)) % k


def oof_predict(spec: Spec, frame: pd.DataFrame, k: int = 4) -> pd.Series:
    f = week_folds(frame.index, k)
    parts = [predict(spec, frame[f != i], frame[f == i]) for i in range(k)]
    return pd.concat(parts).sort_index()


def choose(scores: dict[Spec, float], tol: float = 0.01) -> Spec:
    """主指標最低者；與它相差 tol 以內的改選較簡單的（SIMPLICITY 順序），同結構取 ridge 較大者。"""
    best = min(scores.values())
    near = [s for s, v in scores.items() if v - best <= tol]
    near.sort(key=lambda s: (SIMPLICITY.index((s.seg, s.temp)), -s.ridge))
    return near[0]
