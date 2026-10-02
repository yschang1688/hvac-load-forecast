"""P10 第二段｜OpenEEmeter（CalTRACK 小時模型）對照簡化版基準。

設計在 docs/P10B_PLAN.md 事前登記（先於任何 eemeter 結果）。重點：
- eemeter 4.1.1 的 `HourlyCaltrackModel`：三個月加權分段、佔用分群、溫度分箱用**華氏**，所以外氣溫先換 °F。
- CalTRACK 要求基準跨度剛好 365 天：主比較改從 2024-01-01 起算（P9／P10 從 01-08 起算只有 358 天）。
- 兩種方法吃同一份輸入；自算冷量的負值小時設為缺值。

前置同 run_p10.py（2024 格點資料＋Kaggle 2025 解壓），另需 `uv pip install eemeter`（4.1.1）。
    python src/run_p10b.py
輸出：reports/p10b_monthly.csv、reports/p10b_summary.txt
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mv_baseline as mb  # noqa: E402  （先載入 lightgbm）
import run_p10 as p10  # noqa: E402
from mv_metrics import cv_rmse, nmbe  # noqa: E402
from run_p8 import HU_HOLIDAYS_2024  # noqa: E402

T = lambda x: pd.Timestamp(x, tz=mb.TZ)  # noqa: E731
REPORT_START, REPORT_END, REGIME_BACK = p10.REPORT_START, p10.REPORT_END, p10.REGIME_BACK
EVENT = (T("2024-10-16"), T("2024-11-08"))          # P8 §三 的工況切換事件
TRAP_S = 0.15


# ---------------- 資料 ----------------
def frame_2024_full() -> pd.DataFrame:
    """同 run_p10.frame_2024，但從 2024-01-01 起算（CalTRACK 365 天）。"""
    tw = pd.read_parquet(ROOT / "data/processed/bosch_twins_15min.parquet")
    pf = pd.read_parquet(ROOT / "data/processed/bosch_plant_15min.parquet")
    h = p10.physics_hourly(tw.WC000_flow_m3h, tw.WC000_supply_c, tw.WC000_return_c, pf.outdoor_c, HU_HOLIDAYS_2024)
    return h[h.index < T("2025-01-01")]


def clean(h: pd.DataFrame) -> pd.DataFrame:
    """負值冷量（回水溫低於供水溫）物理上不可能，設為缺值；兩種方法共用。"""
    h = h.copy()
    h.loc[h.kw < 0, "kw"] = np.nan
    return h


# ---------------- eemeter ----------------
def _ee():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import eemeter
        import eemeter.eemeter as ee   # 4.x 的 CalTRACK 類別在子套件，頂層沒有匯出
    ee.__version__ = eemeter.__version__
    return ee


def _to_f(t_c: pd.Series) -> pd.Series:
    return t_c * 9.0 / 5.0 + 32.0


def eemeter_fit(base: pd.DataFrame):
    """回傳 (model, baseline_data)。冷量不是電：is_electricity_data=False（否則 0 會被當缺值）。"""
    ee = _ee()
    b = base.dropna(subset=["kw", "t"])
    data = ee.HourlyCaltrackBaselineData.from_series(b.kw.rename("observed"), _to_f(b.t).rename("temperature"),
                                                     is_electricity_data=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ee.HourlyCaltrackModel().fit(data)
    return model, data


def eemeter_predict(model, rep: pd.DataFrame) -> pd.DataFrame:
    ee = _ee()
    data = ee.HourlyCaltrackReportingData.from_series(rep.kw.rename("observed"), _to_f(rep.t).rename("temperature"),
                                                      is_electricity_data=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model.predict(data)
    out.index = out.index.tz_convert(mb.TZ)
    return out


def eemeter_builder(frame: pd.DataFrame, report_start: pd.Timestamp) -> pd.Series:
    """與 mb.Builder 同介面：報告期逐小時調整後基準（kW）。只用基準期能耗與報告期外氣溫。"""
    model, _ = eemeter_fit(frame[frame.index < report_start])
    rep = frame[frame.index >= report_start]
    return eemeter_predict(model, rep).predicted.reindex(rep.index).rename("adj_baseline")


def towt_builder(frame: pd.DataFrame, report_start: pd.Timestamp) -> pd.Series:
    """簡化版 frozen_towt，先丟掉缺值（design 不收 NaN）。"""
    f = frame.dropna(subset=["kw", "t"])
    return mb.frozen_towt(f, report_start)


# ---------------- 報表 ----------------
def fit_quality(y: pd.Series, p: pd.Series) -> dict:
    j = pd.DataFrame({"y": y, "p": p}).dropna()
    mo = j.resample("MS").sum()
    return {"cv_h": cv_rmse(j.y, j.p), "nmbe_h": nmbe(j.y, j.p), "cv_m": cv_rmse(mo.y, mo.p), "nmbe_m": nmbe(mo.y, mo.p)}


def regime_pct(adj: pd.Series, actual: pd.Series) -> list[tuple[str, float, int]]:
    out = []
    for lab, lo, hi in [("高供水溫工況 01-01～05-10", REPORT_START, REGIME_BACK),
                        ("回到 2024 工況 05-11～05-31", REGIME_BACK, REPORT_END)]:
        j = pd.DataFrame({"a": adj, "y": actual}).dropna()
        j = j[(j.index >= lo) & (j.index < hi)]
        out.append((lab, (j.a.sum() - j.y.sum()) / j.a.sum(), len(j)))
    return out


def main():
    base_full = clean(frame_2024_full())
    rep = clean(p10.frame_2025())
    h = pd.concat([base_full, rep])
    lines = [f"P10 第二段摘要：OpenEEmeter {_ee().__version__} HourlyCaltrackModel 對照簡化版 frozen_towt"]
    lines.append(f"  輸入：WC000 自算冷量；負值設缺值（2024 {int((frame_2024_full().kw < 0).sum())} 小時、"
                 f"2025 {int((p10.frame_2025().kw < 0).sum())} 小時）；報告 2025-01-01～05-31")

    # 充足性警告（M1、M3）
    lines += ["", "## 一、CalTRACK 充足性檢查（eemeter 原樣輸出）"]
    m1_model, m1_data = eemeter_fit(base_full)
    base_0108 = base_full[base_full.index >= T("2024-01-08")]
    m3_model, m3_data = eemeter_fit(base_0108)
    for lab, d in [("M1 基準 2024-01-01～12-31", m1_data), ("M3 基準 2024-01-08～12-31", m3_data)]:
        names = [f"{w.qualified_name.split('.')[-1]} {w.data}" for w in d.warnings]
        lines.append(f"  {lab}：{'無警告' if not names else '；'.join(names)}")

    # 基準期擬合
    lines += ["", "## 二、基準期擬合（樣本內；NMBE 正＝低估；Guideline 14-2002：每小時 30%／±10%、每月 15%／±5%）"]
    b = base_full.dropna(subset=["kw", "t"])
    p_m1 = eemeter_predict(m1_model, b).predicted
    p_m2 = pd.Series(mb.design(b) @ mb.fit_towt(b), index=b.index)
    fq = {}
    for lab, p in [("M1 eemeter", p_m1), ("M2 簡化版", p_m2)]:
        q = fit_quality(b.kw, p)
        fq[lab] = q
        lines.append(f"  {lab}：每小時 CV {q['cv_h']:.1%} NMBE {q['nmbe_h']:+.1%}｜每月 CV {q['cv_m']:.1%} NMBE {q['nmbe_m']:+.1%}")

    # 報告期
    lines += ["", "## 三、跨年未解釋差異（調整後基準 − 實際；負值＝實際比基準多）"]
    base_noevent = base_full.copy()
    ev = (base_noevent.index >= EVENT[0]) & (base_noevent.index < EVENT[1])
    base_noevent.loc[ev, "kw"] = np.nan
    pred_m1 = eemeter_predict(m1_model, rep)
    runs = {
        "M1 eemeter 全年": pred_m1.predicted.reindex(rep.index),
        "M2 簡化版 全年": towt_builder(h, REPORT_START),
        "M3 eemeter 01-08 起": eemeter_predict(m3_model, rep).predicted.reindex(rep.index),
        "M4 eemeter 去事件": eemeter_predict(eemeter_fit(base_noevent)[0], rep).predicted.reindex(rep.index),
    }
    rows = []
    for lab, adj in runs.items():
        t = mb.avoided_table(adj, rep.kw)
        m = t.drop(index="total").avoided_pct
        lines.append(f"  {lab:16s}：" + "　".join(f"{i.strftime('%m')}月 {v:+.1%}" for i, v in m.items())
                     + f"｜合計 {t.loc['total', 'avoided_pct']:+.1%}")
        for lab2, pct, n in regime_pct(adj, rep.kw):
            lines.append(f"  {'':16s}  {lab2}：{n:,} 小時，{pct:+.1%}")
        for i, r in t.iterrows():
            rows.append({"run": lab, "month": "total" if i == "total" else i.strftime("%Y-%m"),
                         "actual_mwh": r.actual, "adj_baseline_mwh": r.adj_baseline, "avoided_pct": r.avoided_pct})

    t1 = mb.avoided_table(runs["M1 eemeter 全年"], rep.kw)
    t2 = mb.avoided_table(runs["M2 簡化版 全年"], rep.kw)
    s1, s2 = np.sign(t1.drop(index="total").avoided_pct), np.sign(t2.drop(index="total").avoided_pct)
    lines.append(f"  同向月份 {int((s1 == s2).sum())}/5；方法差（M1 − M2 合計）"
                 f"{(t1.loc['total', 'avoided_pct'] - t2.loc['total', 'avoided_pct']) * 100:+.1f} 個百分點")

    # 不確定度：predict 給每小時 avg_unc，逐月總不確定度＝sqrt(Σ avg_unc²)
    lines += ["", "## 四、M1 逐月差異 vs eemeter 自己的 90% 不確定度（ASHRAE 14 型，MWh）"]
    j = pred_m1.dropna(subset=["observed", "predicted"])
    unc = (j.predicted_uncertainty ** 2).resample("MS").sum() ** 0.5 / 1000.0
    for i, r in t1.drop(index="total").iterrows():
        u = unc.get(i, np.nan)
        lines.append(f"  {i.strftime('%Y-%m')}：差異 {r.avoided:+7.1f} MWh，不確定度 ±{u:5.1f} MWh，"
                     f"{'超出' if abs(r.avoided) > u else '在內'}")

    # 守門與植入
    lines += ["", f"## 五、守門與植入 {TRAP_S:.0%}（M1）"]
    lines.append(f"  守門（報告期能耗是否流進基準）：{mb.depends_on_reporting_energy(eemeter_builder, h, REPORT_START)}")
    f0 = t1.loc["total", "avoided_pct"]
    hi = mb.inject_savings(h, REPORT_START, TRAP_S)
    f1 = mb.avoided_table(eemeter_builder(hi, REPORT_START), hi.kw[hi.index >= REPORT_START]).loc["total", "avoided_pct"]
    expect = 1 - (1 - TRAP_S) * (1 - f0)
    lines.append(f"  不植入 {f0:+.1%}；植入後 {f1:+.1%}；不吸收時應得 {expect:+.1%}；被吃掉 {expect - f1:+.2%}")

    pd.DataFrame(rows).to_csv(ROOT / "reports/p10b_monthly.csv", index=False)
    txt = "\n".join(lines)
    (ROOT / "reports/p10b_summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
