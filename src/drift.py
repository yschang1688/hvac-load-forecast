"""P4｜漂移監控：PSI 與重訓觸發。

**PSI 的三個實作陷阱**（都會讓監控看起來在運作、實際上是噪音產生器）：
1. 分箱邊界必須來自**參考期**並凍結。用當期資料重新分箱，兩邊分布定義就都變了，
   算出來的 PSI 沒有意義。
2. 空箱要加平滑，否則 log(0) → inf，一根空箱就能讓 PSI 爆表。
   **而且平滑方式不能是「把比例 clip 到某個極小值」**——本專案初版用 `clip(1e-6)`，
   結果每個空箱貢獻約 1.1 的 PSI，週視窗（168 點／10 箱）遇上零膨脹資料常有 3 個以上空箱，
   PSI 於是穩定落在 7–9，**312 個監控週 100% 判 alert**。
   一個永遠 alert 的告警與一個壞掉的告警無法區分，維運會直接忽略它。
   正解是 Laplace 平滑（每箱加 0.5 個計數再正規化），讓空箱的貢獻與樣本數掛鉤。
3. 零膨脹欄位（本資料集的負荷）若用等寬分箱，幾乎所有質量會落在同一箱，
   PSI 對真實漂移不敏感。改用參考期的分位數分箱。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ALPHA = 0.5      # Laplace 平滑的每箱先驗計數


def psi_bins(ref: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """以參考期分位數決定分箱邊界，**一旦決定就凍結**。"""
    r = ref[np.isfinite(ref)]
    if len(r) == 0:
        return np.array([-np.inf, np.inf])
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(r, qs))
    if len(edges) < 3:                       # 退化（例如常數或極端零膨脹）
        return np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def psi(ref: np.ndarray, cur: np.ndarray, edges: np.ndarray) -> float:
    r = ref[np.isfinite(ref)]
    c = cur[np.isfinite(cur)]
    if len(r) == 0 or len(c) == 0 or len(edges) < 3:
        return np.nan
    pr = np.histogram(r, bins=edges)[0].astype(float)
    pc = np.histogram(c, bins=edges)[0].astype(float)
    # Laplace 平滑：每箱加 alpha 個計數再正規化。空箱的機率因此是 alpha/(n+k*alpha)，
    # 隨樣本數縮小而不是固定在一個極小常數——這正是 clip 版本壞掉的地方。
    a = ALPHA
    pr = (pr + a) / (pr.sum() + a * len(pr))
    pc = (pc + a) / (pc.sum() + a * len(pc))
    return float(((pc - pr) * np.log(pc / pr)).sum())


def monitor(ref_X: pd.DataFrame, cur_X: pd.DataFrame, cols: list[str],
            edges: dict[str, np.ndarray]) -> dict[str, float]:
    return {c: psi(ref_X[c].to_numpy(float), cur_X[c].to_numpy(float), edges[c])
            for c in cols if c in cur_X.columns}


def verdict(psi_by_col: dict[str, float], warn: float = 0.10, alert: float = 0.25) -> tuple[str, float]:
    """業界慣例門檻：<0.10 穩定／0.10–0.25 需注意／>0.25 顯著漂移。

    以**最大單欄 PSI** 判定而非平均——平均會被大量穩定欄位稀釋掉一個真的壞掉的特徵。
    """
    vals = [v for v in psi_by_col.values() if np.isfinite(v)]
    if not vals:
        return "unknown", np.nan
    m = max(vals)
    return ("alert" if m > alert else "warn" if m > warn else "stable"), m


def inject_concept_drift(y: pd.Series, start: pd.Timestamp, scale: float = 1.6,
                         shift_hours: int = 4) -> pd.Series:
    """概念漂移注入：從 start 起把負荷型態改掉（放大 + 尖峰時段平移）。

    模擬「大樓使用行為改變」——特徵分布可以幾乎不動（外氣一樣、日曆一樣），
    但特徵與標的之間的關係變了。**這種漂移 PSI 抓不到**，
    因為 PSI 看的是輸入分布，不是輸入到輸出的映射。這正是演練要證明的事。
    """
    out = y.copy()
    m = out.index >= start
    seg = out[m]
    rolled = seg.shift(shift_hours).bfill()
    out.loc[m] = (rolled * scale).to_numpy()
    return out
