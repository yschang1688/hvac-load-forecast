"""節能量測驗證（M&V）常用的兩個模型指標：CV(RMSE) 與 NMBE。

定義依 ASHRAE Guideline 14 的慣用寫法（p＝模型參數數，常取 1）：
  CV(RMSE) = sqrt( Σ(y − ŷ)² / (n − p) ) / ȳ
  NMBE     = Σ(y − ŷ) / ((n − p) · ȳ)
NMBE 為正代表模型**低估**（實際比預測高），為負代表高估。
MAE、RMSE 只看誤差大小，看不到方向；NMBE 看得到系統性偏差，這是 M&V 在意它的原因：
基準模型系統性偏低，算出來的節能量就會系統性偏少（反之偏多），直接影響分潤。

門檻（校準／驗收**基準模型**用，不是日前預測的驗收標準；Guideline 14 原文在付費牆後，
本 repo 未一手核對，數值取自多個二手來源一致的寫法）：
  每小時 CV(RMSE) ≤ 30%、|NMBE| ≤ 10%；每月 CV(RMSE) ≤ 15%、|NMBE| ≤ 5%。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

G14_THRESHOLDS = {"hourly": (0.30, 0.10), "monthly": (0.15, 0.05)}


def _pair(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    return y[ok], yhat[ok]


def cv_rmse(y, yhat, p: int = 1) -> float:
    y, yhat = _pair(y, yhat)
    n = len(y)
    if n <= p or y.mean() == 0:
        return float("nan")
    return float(np.sqrt(((y - yhat) ** 2).sum() / (n - p)) / y.mean())


def nmbe(y, yhat, p: int = 1) -> float:
    y, yhat = _pair(y, yhat)
    n = len(y)
    if n <= p or y.mean() == 0:
        return float("nan")
    return float((y - yhat).sum() / ((n - p) * y.mean()))


def g14_table(df: pd.DataFrame, truth: str, methods: list[str],
              rules: dict[str, str] | None = None) -> pd.DataFrame:
    """把 15 分鐘（或更細）的預測聚合到各時間顆粒（平均功率），逐方法算兩個指標。

    聚合用平均：以 kW 計，平均值 × 時數＝能量，CV 與 NMBE 對常數倍率不變。
    只取所有方法與實際值都有值的時段，避免不同方法用不同樣本比。
    """
    rules = rules or {"hourly": "1h", "daily": "1D", "monthly": "MS"}
    base = df[[truth] + methods].dropna()
    rows = []
    for name, rule in rules.items():
        a = base.resample(rule).mean().dropna()
        for m in methods:
            rows.append({"granularity": name, "method": m, "n": len(a),
                         "cv_rmse": cv_rmse(a[truth], a[m]), "nmbe": nmbe(a[truth], a[m])})
    return pd.DataFrame(rows)
