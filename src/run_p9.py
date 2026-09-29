"""P9｜照 docs/P9_PLAN.md 事前登記的設定執行：安慰劑、植入節能回收、兩個陷阱、守門檢查。

前置：P8 的格點資料（`python src/bosch_plant.py`）。
    python src/run_p9.py        # 約 10 秒
輸出：reports/p9_monthly.csv（逐月明細）、reports/p9_summary.txt
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mv_baseline as mb  # noqa: E402  （先載入 lightgbm）
from mv_metrics import cv_rmse, nmbe  # noqa: E402

A = pd.Timestamp("2024-07-01", tz=mb.TZ)
B = pd.Timestamp("2024-10-01", tz=mb.TZ)
SHARES = (0.05, 0.10, 0.15)
TRAP_S = 0.15


def fit_quality(frame: pd.DataFrame, report_start: pd.Timestamp) -> dict:
    """基準期樣本內擬合的 CV(RMSE)／NMBE，每小時與每月。"""
    base = frame[frame.index < report_start]
    pred = pd.Series(mb.design(base) @ mb.fit_towt(base), index=base.index)
    mo = pd.DataFrame({"y": base.kw, "p": pred}).resample("MS").sum()
    return {"cv_h": cv_rmse(base.kw, pred), "nmbe_h": nmbe(base.kw, pred),
            "cv_m": cv_rmse(mo.y, mo.p), "nmbe_m": nmbe(mo.y, mo.p)}


def run_case(frame, report_start, builder_name, s):
    f = mb.inject_savings(frame, report_start, s)
    adj = mb.BUILDERS[builder_name](f, report_start)
    t = mb.avoided_table(adj, f.kw[f.index >= report_start])
    return t.assign(builder=builder_name, injected=s, report_start=str(report_start.date()))


def main():
    h = mb.hourly_frame()
    rows, lines = [], [f"P9 摘要（每小時 {len(h):,} 列，{h.index[0].date()}～{h.index[-1].date()}）", ""]

    lines.append("## 一、基準期擬合（frozen_towt，樣本內；NMBE 正＝低估；Guideline 14-2002：每小時 30%／±10%、每月 15%／±5%）")
    for lab, rs in [("A", A), ("B", B)]:
        q = fit_quality(h, rs)
        lines.append(f"  {lab}：每小時 CV {q['cv_h']:.1%} NMBE {q['nmbe_h']:+.1%}｜每月 CV {q['cv_m']:.1%} NMBE {q['nmbe_m']:+.1%}")

    lines += ["", "## 二、安慰劑（真實節能＝0，算出來的都是模型誤差）"]
    for lab, rs in [("A", A), ("B", B)]:
        t = run_case(h, rs, "frozen_towt", 0.0)
        rows.append(t.assign(experiment=f"placebo_{lab}"))
        months = t.drop(index="total")
        lines.append(f"  {lab}：合計 {t.loc['total', 'avoided_pct']:+.1%}（{t.loc['total', 'avoided']:+.0f} MWh）；"
                     f"逐月 {months.avoided_pct.min():+.1%} ～ {months.avoided_pct.max():+.1%}")

    lines += ["", "## 三、植入節能回收（frozen_towt，設計 A）"]
    placebo_a = run_case(h, A, "frozen_towt", 0.0)
    for s in SHARES:
        t = run_case(h, A, "frozen_towt", s)
        rows.append(t.assign(experiment=f"recovery_{int(s * 100)}"))
        lines.append(f"  植入 {s:.0%}：算出 {t.loc['total', 'avoided_pct']:+.1%}（安慰劑誤差 {placebo_a.loc['total', 'avoided_pct']:+.1%}）")

    lines += ["", f"## 四、兩個陷阱（設計 A，植入 {TRAP_S:.0%}）；被吃掉＝不吸收時應得值 − 植入後算出值，"
                  "應得值＝1 −（1 − s）×（1 − 同方法安慰劑誤差）"]
    for name in ["frozen_towt", "lag_lgbm", "rolling_towt"]:
        t1 = run_case(h, A, name, TRAP_S)
        t0 = run_case(h, A, name, 0.0)
        rows.append(t1.assign(experiment=f"trap_{name}"))
        rows.append(t0.assign(experiment=f"trap_{name}_placebo"))
        # 植入是乘法：不吸收時，算出值＝1 − (1 − s)·actual/baseline＝1 − (1 − s)(1 − 安慰劑誤差)
        eaten = (1 - (1 - TRAP_S) * (1 - t0.avoided_pct)) - t1.avoided_pct
        per = "  ".join(f"{i.strftime('%m')}月 {v:+.1%}" for i, v in eaten.drop(index="total").items())
        lines.append(f"  {name:13s} 算出 {t1.loc['total', 'avoided_pct']:+.1%}；被吃掉 合計 {eaten['total']:+.1%}｜{per}")

    lines += ["", "## 五、守門：調整後基準是否依賴報告期能耗（True＝會吸收節能）"]
    for name, b in mb.BUILDERS.items():
        lines.append(f"  {name:13s} {mb.depends_on_reporting_energy(b, h, A)}")

    out = pd.concat(rows).reset_index().rename(columns={"index": "month"})
    out.to_csv(ROOT / "reports/p9_monthly.csv", index=False)
    txt = "\n".join(lines)
    (ROOT / "reports/p9_summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
