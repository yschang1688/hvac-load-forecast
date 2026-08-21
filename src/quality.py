"""P1｜多案場資料品質閘：規則登錄化、三級判定、provenance 保留。

設計三條原則（沿用 instrument-data-gateway）：
1. **原始值不刪不改**：閘門只產生判定與標記，修補另存欄位（`value_clean`／`imputed`／`impute_method`）。
2. **三級判定而非二分法**：Pass／Conditional／Reject。Conditional 是「可用但要標註」，
   二分法會逼人把可用資料丟掉、或把髒資料當乾淨。
3. **規則要能說出物理理由**：每條規則附 `rationale`，被質疑時答得出來（JD 原文要求）。

**因果性分層**（時序資料特有，勿混用）：
- `streaming_safe=True` 的規則只看當下與過去，生產環境逐筆進來也能跑。
- `streaming_safe=False` 的規則需要整段序列（如覆蓋率、最長平坦段），只能在批次剖析階段用，
  **其判定結果不得回頭當成特徵**，否則就是 look-ahead。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal
import numpy as np
import pandas as pd

Severity = Literal["ERROR", "WARN"]
Verdict = Literal["Pass", "Conditional", "Reject"]


@dataclass(frozen=True)
class Rule:
    code: str
    name: str
    severity: Severity
    rationale: str
    streaming_safe: bool
    check: Callable[[pd.Series, dict], pd.Series]  # 回傳 bool mask（True＝違規）


RULES: list[Rule] = []


def rule(code, name, severity, rationale, streaming_safe):
    def deco(fn):
        RULES.append(Rule(code, name, severity, rationale, streaming_safe, fn))
        return fn
    return deco


@rule("R001", "負值", "ERROR",
      "冰水系統的瞬時熱量/流量不可能為負；負值代表計量器歸零重置或符號錯誤。",
      streaming_safe=True)
def _negative(s, cfg):
    return s < 0


@rule("R002", "超出物理上限", "ERROR",
      "以該案場自身的穩健尺度設上限（非零讀數的 q3 + k×IQR），超過者多為計量器溢位或單位跳變。"
      "⚠ 分位數必須在**非零子集**上估：冰水資料是零膨脹的（冰機關機時讀數為 0），"
      "本資料集有案場零值佔比達 77–80%，含零計算會讓 q1=q3=0、IQR 塌成 0、上限變成 0，"
      "於是**每一個非零讀數都被判成超出物理上限**——本專案初版即如此，7 個案場貢獻了"
      "R002 全部命中的 62%。這與 R003 的尺度、R004 的平坦段是同一個根因："
      "任何以整體分布估計的統計量，都會被那些代表『正常關機』的 0 拖垮。",
      streaming_safe=False)
def _ceiling(s, cfg):
    v = s.dropna()
    v = v[v > 0]
    if len(v) < cfg.get("min_nonzero_for_ceiling", 100):
        return pd.Series(False, index=s.index)   # 樣本太少不判，不猜
    q1, q3 = v.quantile(.25), v.quantile(.75)
    iqr = q3 - q1
    if iqr <= 0:
        return pd.Series(False, index=s.index)   # 尺度退化就不判，勿用 1e-9 硬撐出一個假上限
    cap = q3 + cfg.get("ceiling_iqr_k", 20) * iqr
    return s > cap


@rule("R003", "突刺", "WARN",
      "單小時變化量超過該案場自身運轉水準的 k 倍。建築有熱慣性，冰水負荷不可能在一小時內"
      "跳掉自身典型運轉量級的數倍；水準以過去 7 天的**非零**中位數估計（shift(1) 後 rolling，"
      "不看未來）。"
      "⚠ 尺度不可用『過去 diff 的中位數』：那會讓門檻隨序列平滑度浮動——夜間平坦的案場 "
      "median|diff| 趨近 0，白天正常爬升就會觸發。本專案初版即如此，實測在 465 棟上"
      "**全部觸發、命中 24 萬點（約 3% 觀測點）**，是誤報產生器而非檢核；"
      "改以運轉水準為尺度後降到約 0.6%。",
      streaming_safe=True)
def _spike(s, cfg):
    d = s.diff().abs()
    w = cfg.get("spike_window", 168)
    level = s.where(s > 0).shift(1).rolling(w, min_periods=24).median()
    k = cfg.get("spike_k", 3)
    return (d > k * level) & level.notna() & (level > 0)


@rule("R004", "非零平坦段（sensor 卡死）", "ERROR",
      "連續 N 小時讀數完全不變且非零＝計量器停止更新。"
      "**必須排除零值**：冰機關機時讀數本來就是 0，把零值平坦段一起抓會製造大量誤報"
      "（本資料集實測：不分零值會抓出 394 棟，分開後真正可疑的只有 117 棟，誤報率約 70%）。",
      streaming_safe=True)
def _flat_nonzero(s, cfg):
    n = cfg.get("flat_hours", 24)
    grp = (s != s.shift()).cumsum()
    run = s.groupby(grp).transform("size")
    return (run >= n) & (s != 0) & s.notna()


@rule("R005", "短缺口（可插補）", "WARN",
      "連續缺值 < 6 小時，建築熱慣性使線性插補在此尺度下可接受；標記後仍可用。",
      streaming_safe=False)
def _gap_short(s, cfg):
    return _gap_mask(s, 1, cfg.get("gap_long_hours", 6) - 1)


@rule("R006", "長缺口（不可插補）", "ERROR",
      "連續缺值 >= 6 小時橫跨日內負荷週期，插補會捏造出不存在的尖峰與谷底；"
      "應留白並由多變量推估處理，或整段排除。",
      streaming_safe=False)
def _gap_long(s, cfg):
    return _gap_mask(s, cfg.get("gap_long_hours", 6), 10**9)


def _gap_mask(s: pd.Series, lo: int, hi: int) -> pd.Series:
    isna = s.isna()
    grp = (isna != isna.shift()).cumsum()
    size = isna.groupby(grp).transform("size")
    return isna & (size >= lo) & (size <= hi)


def evaluate(s: pd.Series, cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """對單一建築的時序跑全部規則，回傳 (逐列違規表, 摘要)。"""
    cfg = cfg or {}
    flags = pd.DataFrame(index=s.index)
    for r in RULES:
        flags[r.code] = r.check(s, cfg).fillna(False).astype(bool)
    by_rule = {r.code: int(flags[r.code].sum()) for r in RULES}
    n_err = sum(by_rule[r.code] for r in RULES if r.severity == "ERROR")
    n_warn = sum(by_rule[r.code] for r in RULES if r.severity == "WARN")
    n = len(s)
    err_frac = n_err / n if n else 0.0
    if err_frac > cfg.get("reject_error_frac", 0.05):
        verdict: Verdict = "Reject"
    elif n_err > 0 or n_warn > 0:
        verdict = "Conditional"
    else:
        verdict = "Pass"
    summary = {"n_rows": n, "n_error": n_err, "n_warn": n_warn,
               "error_frac": err_frac, "verdict": verdict, **by_rule}
    return flags, summary


def clean(s: pd.Series, flags: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """產生修補後欄位，**原始值原封保留**，並記錄補值方法（provenance）。"""
    cfg = cfg or {}
    out = pd.DataFrame({"value_raw": s})
    bad = flags[[r.code for r in RULES if r.severity == "ERROR" and r.code != "R006"]].any(axis=1)
    v = s.mask(bad)                       # ERROR 級讀數視為不可信 → 轉缺值
    out["impute_method"] = ""
    short = _gap_mask(v, 1, cfg.get("gap_long_hours", 6) - 1)
    # ⚠ 不可只靠 interpolate(limit=n)：pandas 的 limit 是「每段最多填 n 格」，
    # 12 小時的長缺口會被填掉前 5 格而不是整段跳過（本專案的守門測試實際抓到過這個 bug）。
    # 正解是先算出整條插補序列，再**只在短缺口的位置**採用它。
    interpolated = v.interpolate(method="linear", limit_area="inside")
    filled = v.copy()
    filled[short] = interpolated[short]
    out["value_clean"] = filled
    out.loc[short & filled.notna(), "impute_method"] = "linear"
    out.loc[bad, "impute_method"] = out.loc[bad, "impute_method"].replace("", "dropped_by_rule")
    out["imputed"] = out["impute_method"].ne("")
    return out
