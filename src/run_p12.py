"""P12｜調校簡化版節能基準：12 個候選、整週交錯四折、選擇規則，全部在 docs/P12_PLAN.md 事前登記。

候選與選擇只用 2024 基準期。OpenDSM 兩個小時模型在同一個四折協定下當對照組（不參與選擇）。
2025 報告期是已經看過的資料，選定之後才算一次，只當描述。

需要 OpenDSM 的獨立環境（見 requirements-opendsm.txt）：
    .venv-opendsm/bin/python src/run_p12.py
輸出：reports/p12_candidates.csv、reports/p12_summary.txt
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
import mv_tuning as mt  # noqa: E402
import run_p10 as p10  # noqa: E402
import run_p10b as b  # noqa: E402
import run_p10c as c  # noqa: E402
from mv_metrics import cv_rmse, nmbe  # noqa: E402

REPORT_START = b.REPORT_START
TRAP_S = 0.15


def quality(y: pd.Series, p: pd.Series) -> dict:
    j = pd.DataFrame({"y": y, "p": p}).dropna()
    mo = j.resample("MS").sum()
    return {"cv_h": cv_rmse(j.y, j.p), "nmbe_h": nmbe(j.y, j.p), "cv_m": cv_rmse(mo.y, mo.p), "nmbe_m": nmbe(mo.y, mo.p), "n": len(j)}


def opendsm_oof(base: pd.DataFrame, kind: str) -> tuple[pd.Series, list[str]]:
    """對照組：同一個四折。回傳樣本外預測與過程紀錄。"""
    f = mt.week_folds(base.index)
    parts, notes = [], []
    for i in range(4):
        tr, te = base[f != i], base[f == i]
        try:
            if kind == "caltrack":
                model, data = c.caltrack_fit(tr)
                w = c._msgs(getattr(data, "warnings", []))
                pred = c.caltrack_predict(model, te).predicted
            else:
                n: list[str] = []
                model, data, n = c.hourly_fit(tr)
                w = c._msgs(getattr(data, "disqualification", [])) + n
                pred = c.hourly_predict(model, te).predicted
            parts.append(pred.reindex(te.index))
            if i == 0:
                notes += [x.split(" {")[0] for x in w]
        except Exception as e:  # 停損：記錄，不影響選擇
            notes.append(f"fold {i} 失敗：{type(e).__name__}: {str(e)[:120]}")
    return (pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)), sorted(set(notes))


def main():
    base = b.clean(b.frame_2024_full()).dropna(subset=["kw", "t"])
    rep = b.clean(p10.frame_2025())
    lines = ["P12 摘要：調校簡化版節能基準（候選與選擇只用 2024 基準期；整週交錯四折）",
             f"  基準期 {base.index[0].date()}～{base.index[-1].date()}，{len(base):,} 小時；"
             f"四折各 {np.bincount(mt.week_folds(base.index)).tolist()} 小時"]

    # ---- 一、C0 重現
    lines += ["", "## 一、C0 樣本內（應重現第二段 M2：每小時 36.9%、每月 21.8%）"]
    q0 = quality(base.kw, mt.predict(mt.C0, base, base))
    lines.append(f"  每小時 CV {q0['cv_h']:.1%} NMBE {q0['nmbe_h']:+.1%}｜每月 CV {q0['cv_m']:.1%} NMBE {q0['nmbe_m']:+.1%}")

    # ---- 二、候選
    lines += ["", "## 二、12 個候選的樣本外表現（主指標：每小時 CV(RMSE)）"]
    rows, oof, scores = [], {}, {}
    for s in mt.ALL_SPECS:
        oof[s] = mt.oof_predict(s, base)
        q = quality(base.kw, oof[s])
        scores[s] = q["cv_h"]
        rows.append({"model": s.name, "seg": s.seg, "temp": s.temp, "ridge": s.ridge, **q})
        lines.append(f"  {s.name:26s} 每小時 CV {q['cv_h']:.1%} NMBE {q['nmbe_h']:+.2%}｜每月 CV {q['cv_m']:.1%} NMBE {q['nmbe_m']:+.2%}")
    chosen = mt.choose(scores)
    best = min(scores, key=scores.get)
    lines.append(f"  主指標最低：{best.name}（{scores[best]:.1%}）；依 1.0 個百分點規則選定：**{chosen.name}**（{scores[chosen]:.1%}）")
    lines.append(f"  C0 {scores[mt.C0]:.1%} → 選定 {scores[chosen]:.1%}，差 {(scores[chosen] - scores[mt.C0]) * 100:+.1f} 個百分點")

    # ---- 三、對照組
    lines += ["", "## 三、對照組：OpenDSM 在同一個四折協定下（不參與選擇）"]
    ctrl = {}
    for kind, lab in [("caltrack", "OpenDSM CalTRACK"), ("hourly", "OpenDSM HourlyModel")]:
        p, notes = opendsm_oof(base, kind)
        ctrl[lab] = p
        if len(p):
            q = quality(base.kw, p)
            rows.append({"model": lab, "seg": "", "temp": "", "ridge": np.nan, **q})
            lines.append(f"  {lab:26s} 每小時 CV {q['cv_h']:.1%} NMBE {q['nmbe_h']:+.2%}｜每月 CV {q['cv_m']:.1%} NMBE {q['nmbe_m']:+.2%}（{q['n']:,} 小時）")
        lines.append(f"  {'':26s} 紀錄：{'；'.join(notes) if notes else '無'}")

    # ---- 四、安慰劑與植入
    lines += ["", "## 四、選定模型與 C0 的安慰劑（留出週沒有任何改善，正確答案 0）與植入 15%"]
    for lab, s in [("C0", mt.C0), ("選定", chosen)]:
        t = mb.avoided_table(oof[s], base.kw)
        m = t.drop(index="total").avoided_pct
        lines.append(f"  {lab}（{s.name}）安慰劑：合計 {t.loc['total', 'avoided_pct']:+.2%}；逐月 {m.min():+.1%}～{m.max():+.1%}；"
                     f"逐月絕對值中位數 {m.abs().median():.1%}")
        j = pd.DataFrame({"p": oof[s], "y": base.kw}).dropna()
        rec = (j.p.sum() - (1 - TRAP_S) * j.y.sum()) / j.p.sum()
        mrec = ((j.p - (1 - TRAP_S) * j.y).resample("MS").sum() / j.p.resample("MS").sum())
        lines.append(f"  {'':4s}植入 {TRAP_S:.0%}：合計算回 {rec:+.1%}；逐月 {mrec.min():+.1%}～{mrec.max():+.1%}")
    h = pd.concat([base, rep])
    lines.append(f"  守門（報告期能耗是否流進基準）：C0 {mb.depends_on_reporting_energy(mt.builder(mt.C0), h.dropna(subset=['t']), REPORT_START)}；"
                 f"選定 {mb.depends_on_reporting_energy(mt.builder(chosen), h.dropna(subset=['t']), REPORT_START)}")

    # ---- 五、2025（已看過，僅描述）
    lines += ["", "## 五、2025 報告期未解釋差異（已看過的資料，僅描述，不是選擇依據）"]
    old = pd.read_csv(ROOT / "reports/p10c_monthly.csv")
    for lab, s in [("C0", mt.C0), ("選定", chosen)]:
        adj = mt.predict(s, base, rep)
        t = mb.avoided_table(adj, rep.kw)
        m = t.drop(index="total").avoided_pct
        lines.append(f"  {lab:4s}{s.name:26s}：" + "　".join(f"{i.strftime('%m')}月 {v:+.1%}" for i, v in m.items())
                     + f"｜合計 {t.loc['total', 'avoided_pct']:+.1%}")
        rows.append({"model": f"2025 {lab} {s.name}", "seg": s.seg, "temp": s.temp, "ridge": s.ridge,
                     "cv_h": np.nan, "nmbe_h": np.nan, "cv_m": np.nan, "nmbe_m": np.nan, "n": len(adj.dropna()),
                     "avoided_pct_2025": t.loc["total", "avoided_pct"]})
    for run in ["N1 CalTRACK 全年", "N2 HourlyModel 全年"]:
        v = old[(old.run == run) & (old.month == "total")].avoided_pct.iloc[0]
        lines.append(f"  參照 {run}（P10 第三段）：合計 {v:+.1%}")

    pd.DataFrame(rows).to_csv(ROOT / "reports/p12_candidates.csv", index=False)
    txt = "\n".join(lines)
    (ROOT / "reports/p12_summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
