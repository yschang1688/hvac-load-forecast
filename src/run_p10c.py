"""P10 第三段｜OpenDSM 1.2.7 重跑節能基準。

設計在 docs/P10C_PLAN.md 事前登記（先於任何 OpenDSM 結果）。重點：
- `eemeter` 套件已改名 OpenDSM；CalTRACK 小時模型的程式碼與 eemeter 4.1.1 逐行相同（N1 對帳）。
- OpenDSM 另有新的 `HourlyModel`（N2–N5）：整年一個彈性網路、時間分群、溫度分箱（華氏）。
- opendsm 1.2.7 在本專案的共用 .venv（Python 3.14）import 失敗，根因是 multimethod 2.1 與 scikit-fda 衝突。
  本檔在獨立環境執行，建法見 requirements-opendsm.txt：
      uv venv .venv-opendsm --python 3.13
      uv pip install --python .venv-opendsm/bin/python -r requirements-opendsm.txt
      .venv-opendsm/bin/python src/run_p10c.py
輸出：reports/p10c_monthly.csv、reports/p10c_summary.txt
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mv_baseline as mb  # noqa: E402
import run_p10 as p10  # noqa: E402
import run_p10b as b  # noqa: E402  （資料、報表函式沿用第二段；它的 eemeter 是延遲載入，這裡不會觸發）

T, EVENT, TRAP_S = b.T, b.EVENT, b.TRAP_S
REPORT_START = b.REPORT_START
SEED = 42
SEEDS = [0, 1, 2, 3, 4]


def _dsm():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import opendsm
        import opendsm.eemeter as ee
    ee.__version__ = opendsm.__version__
    return ee


def _df(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"observed": frame.kw, "temperature": b._to_f(frame.t)})


def _msgs(items) -> list[str]:
    return [f"{w.qualified_name.split('.')[-1]} {w.data}" for w in (items or [])]


# ---------------- N1：OpenDSM 的 CalTRACK 小時模型 ----------------
def caltrack_fit(base: pd.DataFrame):
    ee = _dsm()
    x = base.dropna(subset=["kw", "t"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = ee.HourlyCaltrackBaselineData.from_series(x.kw.rename("observed"), b._to_f(x.t).rename("temperature"),
                                                         is_electricity_data=False)
        model = ee.HourlyCaltrackModel().fit(data)
    return model, data


def caltrack_predict(model, rep: pd.DataFrame) -> pd.DataFrame:
    ee = _dsm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = ee.HourlyCaltrackReportingData.from_series(rep.kw.rename("observed"), b._to_f(rep.t).rename("temperature"),
                                                          is_electricity_data=False)
        out = model.predict(data)
    out.index = out.index.tz_convert(mb.TZ)
    return out


# ---------------- N2–N5：OpenDSM 的新小時模型 ----------------
def hourly_fit(base: pd.DataFrame, seed: int = SEED):
    """回傳 (model, data, notes)。先不帶 ignore_disqualification；被擋下就記錄原因再帶 ignore 重跑。"""
    ee = _dsm()
    notes = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = ee.HourlyBaselineData(_df(base), is_electricity_data=False)
        settings = ee.HourlyNonSolarSettings(seed=seed)
        try:
            model = ee.HourlyModel(settings=settings).fit(data)
        except Exception as e:  # DisqualifiedModelError 等
            notes.append(f"fit 被擋下：{type(e).__name__}: {str(e)[:160]}")
            model = ee.HourlyModel(settings=settings).fit(data, ignore_disqualification=True)
    return model, data, notes


def hourly_predict(model, rep: pd.DataFrame, notes: list | None = None) -> pd.DataFrame:
    ee = _dsm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = ee.HourlyReportingData(_df(rep), is_electricity_data=False)
        try:
            out = model.predict(data)
        except Exception as e:
            if notes is not None:
                notes.append(f"predict 被擋下：{type(e).__name__}: {str(e)[:160]}")
            out = model.predict(data, ignore_disqualification=True)
    out.index = out.index.tz_convert(mb.TZ)
    return out


def hourly_builder(frame: pd.DataFrame, report_start: pd.Timestamp) -> pd.Series:
    """與 mb.Builder 同介面。只用基準期能耗與報告期外氣溫。"""
    model, _, _ = hourly_fit(frame[frame.index < report_start])
    rep = frame[frame.index >= report_start]
    return hourly_predict(model, rep).predicted.reindex(rep.index).rename("adj_baseline")


def _total(adj: pd.Series, actual: pd.Series) -> float:
    return mb.avoided_table(adj, actual).loc["total", "avoided_pct"]


def main():
    ee = _dsm()
    base_full = b.clean(b.frame_2024_full())
    rep = b.clean(p10.frame_2025())
    h = pd.concat([base_full, rep])
    lines = [f"P10 第三段摘要：OpenDSM {ee.__version__}（Python {sys.version.split()[0]}）重跑節能基準",
             "  輸入、報告期、負值處理、°F 換算同第二段；新模型 HourlyNonSolarSettings，主比較 seed=42"]

    # ---- 一、N1 對帳
    lines += ["", "## 一、N1 OpenDSM HourlyCaltrackModel 對第二段 M1（eemeter 4.1.1）逐月對帳"]
    n1_model, n1_data = caltrack_fit(base_full)
    n1 = caltrack_predict(n1_model, rep)
    t_n1 = mb.avoided_table(n1.predicted.reindex(rep.index), rep.kw)
    old = pd.read_csv(ROOT / "reports/p10b_monthly.csv")
    old = old[old.run == "M1 eemeter 全年"].set_index("month").avoided_pct
    worst = 0.0
    for i, r in t_n1.iterrows():
        key = "total" if i == "total" else i.strftime("%Y-%m")
        d = (r.avoided_pct - old[key]) * 100
        worst = max(worst, abs(d))
        lines.append(f"  {key}：N1 {r.avoided_pct:+.2%}　M1 {old[key]:+.2%}　差 {d:+.3f} 個百分點")
    lines.append(f"  最大差 {worst:.3f} 個百分點")

    # ---- 二、充足性
    lines += ["", "## 二、充足性警告與不合格原因（原樣輸出）"]
    n2_model, n2_data, n2_notes = hourly_fit(base_full)
    base_0108 = base_full[base_full.index >= T("2024-01-08")]
    n3_model, n3_data, n3_notes = hourly_fit(base_0108)
    for lab, d, m, notes in [("N1 CalTRACK 01-01 起", n1_data, None, []), ("N2 HourlyModel 01-01 起", n2_data, n2_model, n2_notes),
                             ("N3 HourlyModel 01-08 起", n3_data, n3_model, n3_notes)]:
        w = _msgs(getattr(d, "warnings", []))
        dq = _msgs(getattr(d, "disqualification", [])) + (_msgs(getattr(m, "disqualification", [])) if m is not None else [])
        mw = _msgs(getattr(m, "warnings", [])) if m is not None else []
        lines.append(f"  {lab}")
        lines.append(f"    資料警告：{'無' if not w else '；'.join(w)}")
        lines.append(f"    不合格：{'無' if not dq else '；'.join(dq)}")
        if m is not None:
            lines.append(f"    模型警告：{'無' if not mw else '；'.join(mw)}")
        for n in notes:
            lines.append(f"    {n}")

    # ---- 三、基準期擬合
    lines += ["", "## 三、基準期擬合（樣本內；NMBE 正＝低估；Guideline 14-2002：每小時 30%／±10%、每月 15%／±5%）"]
    bb = base_full.dropna(subset=["kw", "t"])
    fits = {
        "N1 CalTRACK": caltrack_predict(n1_model, bb).predicted,
        "N2 HourlyModel": hourly_predict(n2_model, bb).predicted,
        "M2 簡化版": pd.Series(mb.design(bb) @ mb.fit_towt(bb), index=bb.index),
    }
    for lab, p in fits.items():
        q = b.fit_quality(bb.kw, p.reindex(bb.index))
        lines.append(f"  {lab}：每小時 CV {q['cv_h']:.1%} NMBE {q['nmbe_h']:+.1%}｜每月 CV {q['cv_m']:.1%} NMBE {q['nmbe_m']:+.1%}")

    # ---- 四、報告期
    lines += ["", "## 四、跨年未解釋差異（調整後基準 − 實際；負值＝實際比基準多）"]
    base_noevent = base_full.copy()
    ev = (base_noevent.index >= EVENT[0]) & (base_noevent.index < EVENT[1])
    base_noevent.loc[ev, "kw"] = np.nan
    pred_notes: list[str] = []
    pred_n2 = hourly_predict(n2_model, rep, pred_notes)
    n4_model, n4_data, n4_notes = hourly_fit(base_noevent)
    runs = {
        "N1 CalTRACK 全年": n1.predicted.reindex(rep.index),
        "N2 HourlyModel 全年": pred_n2.predicted.reindex(rep.index),
        "M2 簡化版 全年": b.towt_builder(h, REPORT_START),
        "N3 HourlyModel 01-08 起": hourly_predict(n3_model, rep).predicted.reindex(rep.index),
        "N4 HourlyModel 去事件": hourly_predict(n4_model, rep).predicted.reindex(rep.index),
    }
    rows, tabs = [], {}
    for lab, adj in runs.items():
        t = mb.avoided_table(adj, rep.kw)
        tabs[lab] = t
        m = t.drop(index="total").avoided_pct
        lines.append(f"  {lab:20s}：" + "　".join(f"{i.strftime('%m')}月 {v:+.1%}" for i, v in m.items())
                     + f"｜合計 {t.loc['total', 'avoided_pct']:+.1%}")
        for lab2, pct, n in b.regime_pct(adj, rep.kw):
            lines.append(f"  {'':20s}  {lab2}：{n:,} 小時，{pct:+.1%}")
        for i, r in t.iterrows():
            rows.append({"run": lab, "month": "total" if i == "total" else i.strftime("%Y-%m"),
                         "actual_mwh": r.actual, "adj_baseline_mwh": r.adj_baseline, "avoided_pct": r.avoided_pct})
    for n in pred_notes:
        lines.append(f"  N2 {n}")
    for n in n4_notes:
        lines.append(f"  N4 {n}；不合格原因：{'；'.join(_msgs(getattr(n4_data, 'disqualification', []))) or '未列'}"
                     "（去掉事件後 10、11 月覆蓋率不足，N4 是帶 ignore_disqualification 跑出來的）")
    tot = {k: v.loc["total", "avoided_pct"] for k, v in tabs.items()}
    lines.append(f"  N2 − N1 合計 {(tot['N2 HourlyModel 全年'] - tot['N1 CalTRACK 全年']) * 100:+.1f} 個百分點；"
                 f"N2 − M2 合計 {(tot['N2 HourlyModel 全年'] - tot['M2 簡化版 全年']) * 100:+.1f} 個百分點；"
                 f"N4 − N2 合計 {(tot['N4 HourlyModel 去事件'] - tot['N2 HourlyModel 全年']) * 100:+.1f} 個百分點")

    # ---- 事後：同小時集合（HourlyModel 會內插缺的外氣溫再預測，CalTRACK 不會，兩者涵蓋的小時數不同）
    common = runs["N1 CalTRACK 全年"].notna() & runs["N2 HourlyModel 全年"].notna() & rep.kw.notna()
    lines += ["", "## 四之二（事後）、同小時集合上的合計"]
    lines.append(f"  N1 有預測 {int((runs['N1 CalTRACK 全年'].notna() & rep.kw.notna()).sum()):,} 小時；"
                 f"N2 有預測 {int((runs['N2 HourlyModel 全年'].notna() & rep.kw.notna()).sum()):,} 小時；共同 {int(common.sum()):,} 小時")
    for lab in ["N1 CalTRACK 全年", "N2 HourlyModel 全年", "M2 簡化版 全年", "N4 HourlyModel 去事件"]:
        a = runs[lab][common]
        lines.append(f"  {lab:20s}：{(a.sum() - rep.kw[common].sum()) / a.sum():+.1%}")

    # ---- 五、種子
    lines += ["", "## 五、N5 種子敏感度（HourlyModel，全年基準，合計）"]
    seed_tot = {}
    for s in SEEDS:
        m, _, _ = hourly_fit(base_full, seed=s)
        seed_tot[s] = _total(hourly_predict(m, rep).predicted.reindex(rep.index), rep.kw)
        rows.append({"run": f"N5 seed={s}", "month": "total", "actual_mwh": np.nan, "adj_baseline_mwh": np.nan,
                     "avoided_pct": seed_tot[s]})
    lines.append("  " + "　".join(f"seed {s}：{v:+.2%}" for s, v in seed_tot.items()))
    lines.append(f"  範圍 {(max(seed_tot.values()) - min(seed_tot.values())) * 100:.2f} 個百分點（另 seed 42：{tot['N2 HourlyModel 全年']:+.2%}）")

    # ---- 六、不確定度
    lines += ["", "## 六、N2 逐月差異 vs 模型自己的 90% 不確定度（MWh）"]
    j = pred_n2.dropna(subset=["observed", "predicted"])
    col = "predicted_unc" if "predicted_unc" in j.columns else "predicted_uncertainty"   # 1.2.7 改名為 predicted_unc
    if col in j.columns:
        unc = (j[col] ** 2).resample("MS").sum() ** 0.5 / 1000.0   # 同第二段：逐點不確定度平方和開根號
        for i, r in tabs["N2 HourlyModel 全年"].drop(index="total").iterrows():
            u = unc.get(i, np.nan)
            lines.append(f"  {i.strftime('%Y-%m')}：差異 {r.avoided:+7.1f} MWh，不確定度 ±{u:5.1f} MWh，"
                         f"{'超出' if abs(r.avoided) > u else '在內'}")
    else:
        lines.append(f"  predict 輸出沒有不確定度欄（欄位：{list(j.columns)}）")

    # ---- 七、守門與植入
    lines += ["", f"## 七、守門與植入 {TRAP_S:.0%}（N2）"]
    lines.append(f"  守門（報告期能耗是否流進基準）：{mb.depends_on_reporting_energy(hourly_builder, h, REPORT_START)}")
    f0 = tot["N2 HourlyModel 全年"]
    hi = mb.inject_savings(h, REPORT_START, TRAP_S)
    f1 = _total(hourly_builder(hi, REPORT_START), hi.kw[hi.index >= REPORT_START])
    expect = 1 - (1 - TRAP_S) * (1 - f0)
    lines.append(f"  不植入 {f0:+.1%}；植入後 {f1:+.1%}；不吸收時應得 {expect:+.1%}；被吃掉 {expect - f1:+.2%}")

    pd.DataFrame(rows).to_csv(ROOT / "reports/p10c_monthly.csv", index=False)
    txt = "\n".join(lines)
    (ROOT / "reports/p10c_summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
