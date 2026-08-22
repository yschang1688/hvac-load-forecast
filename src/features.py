"""P2｜防洩漏特徵管線。

**本檔的核心風險是時序洩漏，而它有兩種，第二種最容易被忽略。**

第一種：用未來的負荷算過去的特徵。這種靠 rolling(closed='left') 與 shift 就能防，
        並由 `tests/test_features.py` 的竄改實驗釘死。

第二種：**用「實際的未來天氣」當特徵**。day-ahead 預測在生產環境只拿得到氣象預報，
        不是實測值。離線用實測天氣訓練與評估，等於假設預報零誤差——
        這是離線指標漂亮、上線翻車最常見的來源之一，而且它不會被任何
        shift/rolling 的檢查抓到，因為程式上完全合法。
        本專案把它做成明碼開關 `weather_mode`：
          - "perfect_forecast"：用實測未來天氣（＝完美預報的上界，離線常見做法）
          - "lagged_only"     ：只用預測時點之前的天氣（＝沒有預報可用時的下界）
        兩種都跑，差距就是「預報品質對模型的價值」。單報其中一個都是片面的。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CAL_FEATURES = ["hour_sin", "hour_cos", "dow", "is_weekend", "month_sin", "month_cos"]


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    h, m = idx.hour.to_numpy(), idx.month.to_numpy()
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * h / 24),
        "hour_cos": np.cos(2 * np.pi * h / 24),
        "dow": idx.dayofweek.to_numpy(),
        "is_weekend": (idx.dayofweek.to_numpy() >= 5).astype(int),
        "month_sin": np.sin(2 * np.pi * m / 12),
        "month_cos": np.cos(2 * np.pi * m / 12),
    }, index=idx)


def load_lag_features(y: pd.Series, cfg: dict) -> pd.DataFrame:
    """負荷的落後與滾動統計。全部經 shift(1) 起算——時點 t 的特徵只看得到 t-1 以前。

    因果假設（每個特徵都要答得出「為什麼是它」）：
    - lag_1/2/3    ：熱慣性，前幾小時的負荷是當下負荷最強的預測因子
    - lag_24/48/168：日內、隔日與週節律（辦公建築週間/週末的使用型態差異）
    - roll_mean_24 ：近一日的基準水準，吸收季節與營運狀態的緩慢漂移
    - roll_std_24  ：近一日的波動度，區分穩定運轉與啟停頻繁的時段
    """
    out = {}
    for L in cfg.get("load_lags", [1, 2, 3, 24, 48, 168]):
        out[f"load_lag_{L}"] = y.shift(L)
    base = y.shift(1)
    for w in cfg.get("load_roll_windows", [24, 168]):
        out[f"load_roll_mean_{w}"] = base.rolling(w, min_periods=max(3, w // 4)).mean()
        out[f"load_roll_std_{w}"] = base.rolling(w, min_periods=max(3, w // 4)).std()
    return pd.DataFrame(out, index=y.index)


def weather_features(wx: pd.DataFrame, idx: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """**過去**天氣特徵（永遠因果安全，兩種 mode 都用）。

    因果假設：室外乾球溫度與露點決定顯熱與潛熱負荷，是冷卻負荷的物理驅動量；
    雲量影響日射得熱；風速影響外殼熱傳與滲透。
    """
    cols = cfg.get("weather_cols", ["airTemperature", "dewTemperature", "cloudCoverage", "windSpeed"])
    w = _wx_aligned(wx, idx, cols, cfg)
    out = {}
    for c in cols:
        out[f"{c}_lag1"] = w[c].shift(1)
        out[f"{c}_lag24"] = w[c].shift(24)
    out["airTemp_roll_mean_24"] = w["airTemperature"].shift(1).rolling(24, min_periods=6).mean()
    return pd.DataFrame(out, index=idx)


def _wx_aligned(wx, idx, cols, cfg):
    w = wx.reindex(idx)[cols].astype(float)
    return w.interpolate(limit=cfg.get("weather_interp_limit", 3), limit_area="inside")


def future_weather_block(wx: pd.DataFrame, idx: pd.DatetimeIndex, horizon: int,
                         cfg: dict) -> pd.DataFrame:
    """**未來**天氣區塊：時點 t 這一列放進 t+1 … t+H 的天氣。

    這是本專案唯一被允許看未來的東西，而且是刻意的：day-ahead 預測在生產環境
    確實拿得到氣象預報。但實測天氣 ≠ 預報，**用實測就是假設預報零誤差**，
    量到的準確度是上界不是實際值。

    - `weather_mode="perfect_forecast"`：回傳此區塊（上界）
    - `weather_mode="lagged_only"`      ：回傳空表（無預報可用時的下界）

    兩種都跑，差距＝「氣象預報對這個模型值多少」。只報其中一個都是片面的。
    """
    if cfg.get("weather_mode", "perfect_forecast") != "perfect_forecast":
        return pd.DataFrame(index=idx)
    cols = cfg.get("future_weather_cols", ["airTemperature", "dewTemperature"])
    w = _wx_aligned(wx, idx, cols, cfg)
    out = {}
    for h in range(1, horizon + 1):
        for c in cols:
            out[f"fut_{c}_h{h}"] = w[c].shift(-h)
    return pd.DataFrame(out, index=idx)


def build_design(y: pd.Series, wx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """基礎設計矩陣：**全部因果安全**，任何欄位都不得依賴 t 之後的資料。

    未來天氣不在這裡——它由 `future_weather_block` 單獨產生並明碼標記，
    這樣 `detect_lookahead` 對本函式的輸出就是一條乾淨的紅線：回傳非空即為 bug。
    **不在此處 dropna**，切分之後才丟，避免用未來資訊決定樣本存在與否。
    """
    parts = [calendar_features(y.index), load_lag_features(y, cfg), weather_features(wx, y.index, cfg)]
    if cfg.get("_inject_leak", False):
        parts.append(pd.DataFrame({"LEAK_future_mean24":
                                   y.shift(-24).rolling(24, min_periods=1).mean()}, index=y.index))
    return pd.concat(parts, axis=1)


def build_targets(y: pd.Series, horizon: int) -> pd.DataFrame:
    """多步標的：在時點 t 預測 t+1 … t+H。"""
    return pd.DataFrame({f"y_h{h}": y.shift(-h) for h in range(1, horizon + 1)}, index=y.index)


def detect_lookahead(y: pd.Series, wx: pd.DataFrame, cfg: dict,
                     tamper_frac: float = 0.3) -> list[str]:
    """洩漏偵測：把序列尾端整段竄改，重算特徵，回報哪些欄位的**前段**數值被改變了。

    這是行為檢測不是靜態檢查——它抓的是「實際上有沒有用到未來」，
    不是「程式碼裡有沒有寫 shift」。回傳非空即代表該欄位偷看了未來。

    本函式只作用於 `build_design`，而未來天氣已被移出該函式（見 `future_weather_block`），
    所以這裡的紅線是絕對的：**回傳非空即為 bug**，沒有「這個欄位可以例外」的情況。
    """
    cut = int(len(y) * (1 - tamper_frac))
    y2 = y.copy()
    y2.iloc[cut:] = y2.iloc[cut:] * 7.3 + 1000.0
    wx2 = wx.copy()
    wx2.iloc[cut:] = wx2.iloc[cut:] * 3.1 + 50.0
    a = build_design(y, wx, cfg).iloc[:cut]
    b = build_design(y2, wx2, cfg).iloc[:cut]
    bad = []
    for c in a.columns:
        if not a[c].equals(b[c]):
            if not np.allclose(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float),
                               equal_nan=True, rtol=1e-9, atol=1e-9):
                bad.append(c)
    return bad
