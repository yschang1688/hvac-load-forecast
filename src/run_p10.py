"""P10｜跨年節能基準：2024 全年建基準，套到 2025 年 1–5 月（Kaggle 競賽版）。

設計與登記變更見 docs/P10_PLAN.md。目標是 WC000 自算冷量（流量 × ΔT × 1.163），
因為 Kaggle 版 2025 年移除了冷量感測器、日能量表與 WC003 流量。
2025-01 ～ 05-10 冰水供水溫明顯高於設定點（運轉工況不同），05-11 起回到 2024 年的狀態，兩段分開報告。

前置：
    2024：python src/bosch_plant.py（P8 格點資料）
    2025：Kaggle 競賽資料解壓到 data/interim/bosch2025/<月>/RBHU-2025-<月>/RBHU/...（只需 B205WC*、B106WS01*）
    python src/run_p10.py
輸出：reports/p10_monthly.csv、reports/p10_summary.txt
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mv_baseline as mb  # noqa: E402  （先載入 lightgbm）
import bosch_plant as bp  # noqa: E402
from mv_metrics import cv_rmse, nmbe  # noqa: E402
from run_p8 import HU_HOLIDAYS_2024  # noqa: E402

ROOT25 = str(ROOT / "data/interim/bosch2025/*")
REPORT_START = pd.Timestamp("2025-01-01", tz=mb.TZ)
REPORT_END = pd.Timestamp("2025-06-01", tz=mb.TZ)       # 回水溫只到 2025-05
REGIME_BACK = pd.Timestamp("2025-05-11", tz=mb.TZ)      # 供水溫回到 2024 工況的第一週（事前登記）
# 匈牙利 2025 國定假日（報告期內）：1/1、3/15、4/18 耶穌受難日、4/21 復活節週一、5/1；5/2 橋接日未一手核對
HU_HOLIDAYS_2025 = pd.to_datetime(["2025-01-01", "2025-03-15", "2025-04-18", "2025-04-21",
                                   "2025-05-01", "2025-05-02"]).date
TRAP_S = 0.15


def physics_hourly(flow, supply, ret, outdoor, holidays) -> pd.DataFrame:
    kw = bp.physics_load_kw(flow, supply, ret)
    d = pd.DataFrame({"kw": kw, "t": outdoor.reindex(kw.index)})
    d.index = d.index.tz_convert(mb.TZ)
    h = d.resample("1h").mean()
    h["holiday"] = np.isin(h.index.date, holidays).astype(float)
    return h


def frame_2024() -> pd.DataFrame:
    tw = pd.read_parquet(ROOT / "data/processed/bosch_twins_15min.parquet")
    pf = pd.read_parquet(ROOT / "data/processed/bosch_plant_15min.parquet")
    h = physics_hourly(tw.WC000_flow_m3h, tw.WC000_supply_c, tw.WC000_return_c, pf.outdoor_c, HU_HOLIDAYS_2024)
    return h[(h.index >= pd.Timestamp("2024-01-08", tz=mb.TZ)) & (h.index < pd.Timestamp("2025-01-01", tz=mb.TZ))]


def frame_2025() -> pd.DataFrame:
    g = {k: bp.hold_grid(bp.load_point(o, ROOT25), "15min", "1h") for k, o in bp.PLANT.items()
         if k in ("supply_c", "return_c", "flow_m3h")}
    t = bp.load_point(bp.WEATHER["outdoor_c"], ROOT25)
    t = bp.hold_grid(t.mask(bp.flag_sentinel(t)), "15min", "2h")     # −40 °C 哨兵值 2025 年也有
    h = physics_hourly(g["flow_m3h"], g["supply_c"], g["return_c"], t, HU_HOLIDAYS_2025)
    return h[(h.index >= REPORT_START) & (h.index < REPORT_END)]


def main():
    base, rep = frame_2024().dropna(), frame_2025().dropna()
    h = pd.concat([base, rep])
    lines = [f"P10 摘要：基準 {base.index[0].date()}～{base.index[-1].date()}（{len(base):,} 小時），"
             f"報告 {rep.index[0].date()}～{rep.index[-1].date()}（{len(rep):,} 小時）；目標＝WC000 自算冷量", ""]

    pred = pd.Series(mb.design(base) @ mb.fit_towt(base), index=base.index)
    mo = pd.DataFrame({"y": base.kw, "p": pred}).resample("MS").sum()
    lines.append("## 一、基準期擬合（樣本內；NMBE 正＝低估；Guideline 14-2002：每小時 30%／±10%、每月 15%／±5%）")
    lines.append(f"  每小時 CV {cv_rmse(base.kw, pred):.1%} NMBE {nmbe(base.kw, pred):+.1%}｜"
                 f"每月 CV {cv_rmse(mo.y, mo.p):.1%} NMBE {nmbe(mo.y, mo.p):+.1%}")
    t25, t24 = rep.t, base.t
    lines.append(f"  報告期外氣溫 {t25.min():.1f}～{t25.max():.1f} °C，落在基準期 {t24.min():.1f}～{t24.max():.1f} °C 之內："
                 f"{bool(t25.min() >= t24.min() and t25.max() <= t24.max())}")

    adj = mb.frozen_towt(h, REPORT_START)
    tab = mb.avoided_table(adj, rep.kw)
    lines += ["", "## 二、跨年未解釋差異（避免能耗＝調整後基準 − 實際；負值＝實際比基準多）"]
    for i, r in tab.iterrows():
        lab = "合計" if i == "total" else i.strftime("%Y-%m")
        lines.append(f"  {lab}：實際 {r.actual:7.1f} MWh、調整後基準 {r.adj_baseline:7.1f} MWh、差異 {r.avoided_pct:+.1%}")
    seg = []
    for lab, lo, hi in [("高供水溫工況 01-01～05-10", REPORT_START, REGIME_BACK),
                        ("回到 2024 工況 05-11～05-31", REGIME_BACK, REPORT_END)]:
        a = adj[(adj.index >= lo) & (adj.index < hi)]
        y = rep.kw.reindex(a.index)
        pct = (a.sum() - y.sum()) / a.sum()
        seg.append((lab, pct, len(a)))
        lines.append(f"  {lab}：{len(a):,} 小時，差異 {pct:+.1%}")

    lines += ["", f"## 三、植入 {TRAP_S:.0%} 回收與守門（凍結基準）"]
    f0 = mb.avoided_table(mb.frozen_towt(h, REPORT_START), rep.kw).loc["total", "avoided_pct"]
    hi = mb.inject_savings(h, REPORT_START, TRAP_S)
    f1 = mb.avoided_table(mb.frozen_towt(hi, REPORT_START), hi.kw[hi.index >= REPORT_START]).loc["total", "avoided_pct"]
    expect = 1 - (1 - TRAP_S) * (1 - f0)
    lines.append(f"  不植入 {f0:+.1%}；植入後 {f1:+.1%}；不吸收時應得 {expect:+.1%}；被吃掉 {expect - f1:+.2%}")
    lines.append(f"  守門（報告期能耗是否流進基準）：{mb.depends_on_reporting_energy(mb.frozen_towt, h, REPORT_START)}")

    lines += ["", "## 四、敏感度分析（事後）：固定報告期，只換基準期"]
    T = lambda x: pd.Timestamp(x, tz=mb.TZ)  # noqa: E731
    event = (base.index >= T("2024-10-16")) & (base.index < T("2024-11-08"))
    variants = [
        ("2024 全年（主分析）", base),
        ("2024 上半年 1–6 月", base[base.index < T("2024-07-01")]),
        ("2024 下半年 7–12 月", base[base.index >= T("2024-07-01")]),
        ("2024 同季 1–5 月", base[base.index < T("2024-06-01")]),
        ("全年去掉 10/16–11/7 事件", base[~event]),
        ("全年去掉 12 月", base[base.index < T("2024-12-01")]),
        ("全年去掉事件與 12 月", base[~event & (base.index < T("2024-12-01"))]),
    ]
    sens = []
    for lab, b in variants:
        a = pd.Series(mb.design(rep) @ mb.fit_towt(b), index=rep.index)
        t = mb.avoided_table(a, rep.kw)
        m = t.drop(index="total").avoided_pct
        sens.append({"baseline": lab, "hours": len(b), "min": m.min(), "max": m.max(),
                     "total": t.loc["total", "avoided_pct"], "mean_abs": m.abs().mean()})
        lines.append(f"  {lab:18s} 基準 {len(b):5d} 小時｜逐月 {m.min():+.1%} ～ {m.max():+.1%}｜"
                     f"合計 {t.loc['total', 'avoided_pct']:+.1%}｜逐月絕對平均 {m.abs().mean():.1%}")
    pd.DataFrame(sens).to_csv(ROOT / "reports/p10_sensitivity.csv", index=False)

    out = tab.reset_index().rename(columns={"index": "month"})
    out.to_csv(ROOT / "reports/p10_monthly.csv", index=False)
    txt = "\n".join(lines)
    (ROOT / "reports/p10_summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
