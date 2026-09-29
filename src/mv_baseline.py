"""P9｜節能基準模型：調整後基準、安慰劑測試、植入節能回收，以及兩個會吃掉節能的錯誤。

避免能耗＝調整後基準能耗 − 報告期實際能耗。節能量是模型算出來的反事實，
所以基準模型的第一條紀律是：**調整後基準只能由獨立變數（天氣、時間、假日）決定，不得依賴報告期的能耗。**
報告期的能耗已經是改善後的數字，只要它以任何路徑流進基準，節能就會被吸收。
`depends_on_reporting_energy` 把這條紀律寫成行為檢查（同 P2 `detect_lookahead` 的做法：竄改後看輸出有沒有變）。

三種基準做法，介面一致：`builder(frame, report_start) -> pd.Series`（報告期逐小時的調整後基準，kW）
- `frozen_towt`   ：時段×溫度迴歸，只吃獨立變數，基準期訓練後凍結（正確做法）
- `lag_lgbm`      ：LightGBM 加前 1／2／3／24 小時能耗（預測模型的好習慣，放進基準就是錯）
- `rolling_towt`  ：同 frozen_towt，但每月用最近 6 個月重訓（重訓窗口逐月混入改善後資料）

資料邊界：Bosch B205 冰水機房的**冷量**（熱能），不是電量；本模組不掛任何 IPMVP 量測選項名稱。
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import lightgbm as lgb  # noqa: E402  先於其他 OpenMP 使用者載入（見 models.py 頂端註解）
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TZ = "Europe/Budapest"
KNOTS = (5.0, 10.0, 15.0, 20.0, 25.0)       # 外氣溫分段點（°C），事前登記
RIDGE = 1e-3
LAGS = (1, 2, 3, 24)
LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50,
                   subsample=0.9, subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=1,
                   random_state=7)          # 沿用 P8 在 3 月 fold 凍結的超參

Builder = Callable[[pd.DataFrame, pd.Timestamp], pd.Series]


# ---------------- 資料 ----------------
def hourly_frame(holidays=None) -> pd.DataFrame:
    """P8 的 15 分鐘格點 → 每小時：kw（WC003 冷量）、t（外氣溫）、holiday。當地時間索引。"""
    if holidays is None:
        from run_p8 import HU_HOLIDAYS_2024 as holidays
    tw = pd.read_parquet(ROOT / "data/processed/bosch_twins_15min.parquet")
    pf = pd.read_parquet(ROOT / "data/processed/bosch_plant_15min.parquet")
    d = pd.DataFrame({"kw": tw.WC003_load_kw_sensor, "t": pf.outdoor_c.reindex(tw.index)})
    d.index = d.index.tz_convert(TZ)
    h = d.resample("1h").mean()
    h = h[(h.index >= pd.Timestamp("2024-01-08", tz=TZ)) & (h.index < pd.Timestamp("2025-01-01", tz=TZ))]
    h["holiday"] = np.isin(h.index.date, holidays).astype(float)
    return h.dropna()


def inject_savings(frame: pd.DataFrame, report_start: pd.Timestamp, s: float) -> pd.DataFrame:
    """報告期實際能耗乘以 (1 − s)：模擬一個真的省了 s 的改善措施。"""
    f = frame.copy()
    f.loc[f.index >= report_start, "kw"] *= (1 - s)
    return f


# ---------------- 時段×溫度迴歸 ----------------
def design(frame: pd.DataFrame) -> np.ndarray:
    """只由獨立變數組成：星期幾×小時、外氣溫分段線性、假日。**沒有任何能耗欄位。**"""
    tow = (frame.index.dayofweek * 24 + frame.index.hour).to_numpy()
    t = frame.t.to_numpy(dtype=float)
    return np.hstack([np.eye(168)[tow],
                      np.column_stack([t] + [np.maximum(t - k, 0.0) for k in KNOTS]),
                      frame[["holiday"]].to_numpy(dtype=float)])


def fit_towt(frame: pd.DataFrame) -> np.ndarray:
    X, y = design(frame), frame.kw.to_numpy(dtype=float)
    return np.linalg.solve(X.T @ X + RIDGE * np.eye(X.shape[1]), X.T @ y)


def frozen_towt(frame: pd.DataFrame, report_start: pd.Timestamp) -> pd.Series:
    base = frame[frame.index < report_start]
    rep = frame[frame.index >= report_start]
    return pd.Series(design(rep) @ fit_towt(base), index=rep.index, name="adj_baseline")


def rolling_towt(frame: pd.DataFrame, report_start: pd.Timestamp, months: int = 6) -> pd.Series:
    """陷阱二：每月用最近 `months` 個月重訓。報告期開始後，重訓窗口會逐月混入改善後的能耗。"""
    rep = frame[frame.index >= report_start]
    out = []
    for ms in pd.date_range(report_start, rep.index[-1], freq="MS", tz=TZ):
        win = frame[(frame.index >= ms - pd.DateOffset(months=months)) & (frame.index < ms)]
        cur = rep[(rep.index >= ms) & (rep.index < ms + pd.DateOffset(months=1))]
        if len(cur):
            out.append(pd.Series(design(cur) @ fit_towt(win), index=cur.index))
    return pd.concat(out).rename("adj_baseline")


# ---------------- 陷阱一：滯後能耗 ----------------
def _lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({"tow": frame.index.dayofweek * 24 + frame.index.hour,
                      "t": frame.t, "holiday": frame.holiday}, index=frame.index)
    for L in LAGS:
        X[f"kw_lag{L}"] = frame.kw.shift(L)
    return X


def lag_lgbm(frame: pd.DataFrame, report_start: pd.Timestamp) -> pd.Series:
    """預測模型的好習慣放進基準：前幾小時的**實際**能耗在報告期已是改善後的數字。"""
    X = _lag_features(frame)
    tr = (frame.index < report_start) & X.notna().all(axis=1).to_numpy()
    m = lgb.LGBMRegressor(**LGBM_PARAMS).fit(X[tr], frame.kw[tr])
    rep = X[frame.index >= report_start]
    return pd.Series(m.predict(rep), index=rep.index, name="adj_baseline")


BUILDERS: dict[str, Builder] = {"frozen_towt": frozen_towt, "lag_lgbm": lag_lgbm, "rolling_towt": rolling_towt}


# ---------------- 守門與計帳 ----------------
def depends_on_reporting_energy(builder: Builder, frame: pd.DataFrame,
                                report_start: pd.Timestamp) -> bool:
    """行為檢查：竄改報告期實際能耗，調整後基準若有任何一點改變，就代表節能會被吸收。"""
    a = builder(frame, report_start)
    f2 = frame.copy()
    f2.loc[f2.index >= report_start, "kw"] = f2.loc[f2.index >= report_start, "kw"] * 3.1 + 500.0
    b = builder(f2, report_start).reindex(a.index)
    return not np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True)


def avoided_table(adj_baseline: pd.Series, actual: pd.Series) -> pd.DataFrame:
    """逐月與合計的避免能耗（MWh 與占調整後基準的比例）。只用兩者都有值的小時。"""
    j = pd.DataFrame({"adj_baseline": adj_baseline, "actual": actual}).dropna()
    m = j.resample("MS").sum() / 1000.0
    m.loc["total"] = m.sum()
    m["avoided"] = m.adj_baseline - m.actual
    m["avoided_pct"] = m.avoided / m.adj_baseline
    return m
