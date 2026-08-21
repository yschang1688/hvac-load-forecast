"""P2｜資料組裝與 rolling-origin 切分。

切分紀律：**禁用隨機切分**。時序的隨機切分會讓同一天的鄰近小時同時落在訓練與測試，
量到的是內插不是預測（與 surface-defect 的分組洩漏同構，只是換一個維度）。
本檔採 rolling-origin（前推式）：訓練段永遠在驗證段之前，且兩者之間留一段
**gap = horizon**，避免訓練集尾端的標的與驗證集起點重疊。
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

import features

ROOT = Path(__file__).resolve().parents[1]


def load_cfg() -> dict:
    return json.loads((ROOT / "config" / "pipeline.json").read_text(encoding="utf-8"))


def load_sources():
    raw = ROOT / "data" / "raw"
    cw = pd.read_csv(raw / "chilledwater.csv", parse_dates=["timestamp"]).set_index("timestamp")
    wx = pd.read_csv(raw / "weather.csv", parse_dates=["timestamp"])
    return cw, wx


def site_weather(wx: pd.DataFrame, site_id: str) -> pd.DataFrame:
    w = wx[wx.site_id == site_id].drop(columns=["site_id"]).set_index("timestamp").sort_index()
    return w[~w.index.duplicated(keep="first")]


def select_buildings(cfg: dict) -> list[str]:
    """建模樣本選取。規則寫死在設定檔並在此處執行，**不看模型結果再挑**。

    以每個 site 取品質最好的 k 棟做分層抽樣，讓「跨案場」不是只跨到同一個場域。
    """
    v = pd.read_csv(ROOT / "reports" / "p1_quality_verdicts.csv")
    m = cfg["modeling"]
    ok = v[(v.verdict != "Reject") & (v.coverage >= m["min_coverage"])].copy()
    ok["err_rate"] = ok.error_frac
    ok = ok.sort_values(["site_id", "err_rate", "coverage"], ascending=[True, True, False])
    picked = ok.groupby("site_id").head(m["per_site"]).head(m["n_buildings"])
    return picked.building_id.tolist()


@dataclass
class Split:
    name: str
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def rolling_origin_splits(idx: pd.DatetimeIndex, cfg: dict) -> list[Split]:
    m = cfg["modeling"]
    H = m["horizon"]
    start = pd.Timestamp(m["train_start"])
    first_end = pd.Timestamp(m["first_train_end"])
    step = pd.Timedelta(days=m["valid_days"])
    out, k, end = [], 0, first_end
    while True:
        vs = end + pd.Timedelta(hours=H)          # gap = horizon，避免標的重疊
        ve = vs + step
        if ve > idx.max():
            break
        out.append(Split(f"fold{k}", end, vs, ve))
        end = end + step
        k += 1
        if k >= m["max_folds"]:
            break
    assert out, "rolling-origin 切不出任何 fold，檢查 train_start / first_train_end / valid_days"
    _assert_no_overlap(out, H)
    _assert_sealed_untouched(out, cfg)
    return out


def _assert_sealed_untouched(splits: list[Split], cfg: dict):
    """封存段守衛：模型選擇用的任何 fold 都不得延伸進封存測試期。

    封存段只在全部選擇動作結束後開封一次。選擇過程碰過的資料，
    再拿來當「未見過的測試集」就只是自我確認（surface-defect 的 SealedTest 同紀律）。
    """
    sealed = pd.Timestamp(cfg["modeling"]["sealed_test_start"])
    for s in splits:
        assert s.valid_end <= sealed, f"{s.name} 的驗證段伸進封存測試期（{sealed.date()} 之後）"


def _assert_no_overlap(splits: list[Split], horizon: int):
    for s in splits:
        assert s.train_end < s.valid_start, f"{s.name} 訓練段與驗證段相接，沒有留 gap"
        assert (s.valid_start - s.train_end) >= pd.Timedelta(hours=horizon), \
            f"{s.name} 的 gap 小於 horizon，訓練尾端的標的會伸進驗證段"


def sealed_test_window(cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """封存測試期。呼叫本函式即代表要開封，理由請寫進報告。"""
    return pd.Timestamp(cfg["modeling"]["sealed_test_start"]), pd.Timestamp("2018-01-01")


def build_building(bid: str, cw: pd.DataFrame, wx: pd.DataFrame, cfg: dict):
    """單棟建築 → (基礎特徵 X, 未來天氣區塊 F, 多步標的 Y, 清洗後負荷 y)。"""
    import quality
    s = cw[bid]
    flags, _ = quality.evaluate(s, cfg["quality"])
    cleaned = quality.clean(s, flags, cfg["quality"])
    y = cleaned["value_clean"]
    w = site_weather(wx, bid.split("_")[0])
    fcfg = {**cfg["features"], "weather_mode": cfg["modeling"]["weather_mode"]}
    X = features.build_design(y, w, fcfg)
    F = features.future_weather_block(w, y.index, cfg["modeling"]["horizon"], fcfg)
    Y = features.build_targets(y, cfg["modeling"]["horizon"])
    return X, F, Y, y
