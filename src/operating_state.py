"""P7｜冰機運轉狀態的近似：以「連續零值段」判定關機。

**為什麼需要它**：P3 的主結論被 45 個評估視窗中的 2 個翻轉，兩個都是冰機實質關機的冬季視窗
（零值佔比 85–93%）。在那種視窗上，任何正規化子（nMAE、skill）都會退化；
P3 報告的處方是「把關機視窗在聚合前排除，而且排除規則要事前宣告」。本檔就是那條規則。

**為什麼是連續零值段而不是零值佔比**：夜間與週末關機是正常運轉的一部分——
辦公建築一週有一半時間是 0 並不代表「冰機停了」。區分兩者的是**持續時間**：
正常的夜間關機不超過十幾小時，季節性停機是以天計。所以規則是：
屬於「長度 ≥ min_zero_run_hours 的連續零值段」的時數，佔視窗有效時數的比例超過
max_shutdown_share，才判為關機視窗。

**這是近似**。真實案場應以 BMS 的冰機啟停點位為準；BDG2 沒有這個欄位。

**事前宣告的誠實邊界**：預設門檻（24 小時、50%）是在**看過 P3 結果之後**寫下的，
所以用它重新聚合 P3 仍屬敏感度分析，不是事前登錄的檢定。這條規則從本 commit 起凍結，
對**之後**才打開的資料（P3 封存測試期 2017-07 起、Seq2Seq 重跑、線上週監控）才算事前宣告。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

DEFAULTS = {"min_zero_run_hours": 24, "max_shutdown_share": 0.5, "min_observed_frac": 0.5}


def zero_run_mask(y: pd.Series, min_hours: int = DEFAULTS["min_zero_run_hours"]) -> pd.Series:
    """回傳布林序列：該時點是否位於「長度 ≥ min_hours 的連續零值段」內。

    缺值（NaN）會**中斷**零值段——不知道的時段不能算成關機，否則斷訊會被誤判為停機。
    """
    v = y.to_numpy(dtype=float)
    z = v == 0                                   # NaN == 0 為 False，天然中斷
    out = np.zeros(len(v), dtype=bool)
    i = 0
    while i < len(v):
        if z[i]:
            j = i
            while j < len(v) and z[j]:
                j += 1
            if j - i >= min_hours:
                out[i:j] = True
            i = j
        else:
            i += 1
    return pd.Series(out, index=y.index)


def shutdown_share(y: pd.Series, min_hours: int = DEFAULTS["min_zero_run_hours"],
                   context: pd.Series | None = None) -> float:
    """視窗內屬於長零值段的時數 ÷ 視窗有效（非缺值）時數。

    `context` 是包含此視窗、且前後各延伸至少 min_hours 的較長序列。
    **沒有 context 時，跨視窗邊界的零值段會被截斷而低估**——一段從上週延續過來的
    停機，在本週視窗裡可能只剩 20 小時，單看本週會判成正常。
    """
    if context is not None:
        mask = zero_run_mask(context, min_hours).reindex(y.index).fillna(False).astype(bool)
    else:
        mask = zero_run_mask(y, min_hours)
    observed = y.notna()
    n = int(observed.sum())
    return float((mask & observed).sum() / n) if n else np.nan


def window_state(y: pd.Series, cfg: dict | None = None, context: pd.Series | None = None) -> tuple[str, float]:
    """回 (state, share)。state ∈ {"shutdown", "operating", "unknown"}。

    有效時數不足（斷訊為主的視窗）回 unknown——不假裝判定，同 PSI 退化序列的原則。
    """
    c = {**DEFAULTS, **(cfg or {})}
    if len(y) == 0 or y.notna().mean() < c["min_observed_frac"]:
        return "unknown", np.nan
    s = shutdown_share(y, c["min_zero_run_hours"], context)
    return ("shutdown" if s > c["max_shutdown_share"] else "operating"), s
