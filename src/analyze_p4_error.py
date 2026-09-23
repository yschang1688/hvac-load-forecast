"""P6｜以 P4 已落盤的 624 個監控週，離線重算「誤差監控」對「PSI 監控」的鑑別力。

不重跑部署模擬（單棟最長 51 分鐘）——p4_drift.csv 已含每週凍結模型的 skill，
足以回答「若當初用誤差當主訊號，誤報率與偵測率各是多少」。

**門檻與基準期在本檔頂端事前宣告**，不看結果調：
- 基準期＝各棟部署前 BURN_IN 週的凍結組 skill 中位數（凍結後不更新）
- alert＝當週 skill > 基準 × 1.3（drift.ERROR_RATIO_ALERT）
- 誤報率＝注入組**注入前**（且基準期之後）被判 alert 的週佔比
- 偵測率＝注入組**注入後**被判 alert 的週佔比
P4 報告 §三的 27%／51% 用的是「注入前全部週的中位數」當基準（in-sample）；
本檔另報 burn-in 版本，兩者都列，差距就是基準期選法的代價。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drift  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BURN_IN = 8                      # 週；事前宣告
INJECT_START = pd.Timestamp("2017-07-01")


def rates(d: pd.DataFrame, baseline_fn) -> dict:
    fp, det, unk = [], [], 0
    for bid, g in d.groupby("building_id"):
        g = g.sort_values("week")
        base = baseline_fn(g)
        for _, r in g.iterrows():
            st, _ = drift.error_verdict(r.skill_frozen, base)
            if st == "unknown":
                unk += 1; continue
            (det if r.week >= INJECT_START else fp).append(st == "alert")
    return {"誤報率(注入前)": np.mean(fp) if fp else np.nan, "n_pre": len(fp),
            "偵測率(注入後)": np.mean(det) if det else np.nan, "n_post": len(det), "unknown": unk}


def main():
    f = ROOT / "reports" / "p4_drift.csv"
    d = pd.read_csv(f, parse_dates=["week"])
    inj = d[d.inject].dropna(subset=["skill_frozen"])
    print(f"=== P6 誤差監控 vs PSI（注入組 {inj.building_id.nunique()} 棟 × {inj.week.nunique()} 週）===")
    print(f"門檻：skill > 基準 × {drift.ERROR_RATIO_ALERT}；基準期 burn-in {BURN_IN} 週\n")

    psi_alert = (inj.state == "alert")
    print(f"PSI：注入前 alert {psi_alert[inj.week < INJECT_START].mean():.0%}，"
          f"注入後 alert {psi_alert[inj.week >= INJECT_START].mean():.0%} → 恆為 alert，無資訊")

    r_in = rates(inj, lambda g: drift.error_baseline(g[g.week < INJECT_START].skill_frozen))
    print(f"誤差（P4 §三 口徑，基準＝注入前全部週中位數，in-sample）："
          f"誤報 {r_in['誤報率(注入前)']:.0%} (n={r_in['n_pre']})，偵測 {r_in['偵測率(注入後)']:.0%} (n={r_in['n_post']})")

    # burn-in 版：基準只用前 BURN_IN 週，評估只算 BURN_IN 之後的週
    fp, det, unk = [], [], 0
    for bid, g in inj.groupby("building_id"):
        g = g.sort_values("week")
        base = drift.error_baseline(g.skill_frozen.iloc[:BURN_IN])
        for _, r in g.iloc[BURN_IN:].iterrows():
            st, _ = drift.error_verdict(r.skill_frozen, base)
            if st == "unknown":
                unk += 1; continue
            (det if r.week >= INJECT_START else fp).append(st == "alert")
    print(f"誤差（burn-in {BURN_IN} 週凍結基準，out-of-sample）："
          f"誤報 {np.mean(fp):.0%} (n={len(fp)})，偵測 {np.mean(det):.0%} (n={len(det)})，unknown {unk}")

    print("\n逐棟（burn-in 版）：")
    for bid, g in inj.groupby("building_id"):
        g = g.sort_values("week")
        base = drift.error_baseline(g.skill_frozen.iloc[:BURN_IN])
        tail = g.iloc[BURN_IN:]
        pre = [drift.error_verdict(v, base)[0] == "alert" for v in tail[tail.week < INJECT_START].skill_frozen]
        post = [drift.error_verdict(v, base)[0] == "alert" for v in tail[tail.week >= INJECT_START].skill_frozen]
        print(f"  {bid:28s} 基準 {base:.3f}  誤報 {np.mean(pre):.0%}  偵測 {np.mean(post):.0%}")

    # 滾動版：基準＝前 BURN_IN 週（不含當週）的中位數，逐週前推。
    # 事前宣告的取捨：能跟上季節，但緩慢漂移會被基準「吸收」而漏報；抓的是突變。
    fp, det = [], []
    for bid, g in inj.groupby("building_id"):
        g = g.sort_values("week").reset_index(drop=True)
        for i in range(BURN_IN, len(g)):
            base = drift.error_baseline(g.skill_frozen.iloc[i - BURN_IN:i])
            st, _ = drift.error_verdict(g.skill_frozen.iloc[i], base)
            if st == "unknown":
                continue
            (det if g.week.iloc[i] >= INJECT_START else fp).append(st == "alert")
    print(f"\n誤差（滾動 {BURN_IN} 週基準，lag 1）：誤報 {np.mean(fp):.0%} (n={len(fp)})，偵測 {np.mean(det):.0%} (n={len(det)})")
    print("  ↑ 注入是一次性突變（2017-07-01 起放大 1.6×），滾動基準會在數週後把它吸收成新常態，"
          "所以偵測率讀法是「注入後前幾週有沒有抓到」，不是整段佔比。")
    first = []
    for bid, g in inj.groupby("building_id"):
        g = g.sort_values("week").reset_index(drop=True)
        post = g[g.week >= INJECT_START]
        hit = None
        for i in post.index:
            base = drift.error_baseline(g.skill_frozen.iloc[i - BURN_IN:i])
            if drift.error_verdict(g.skill_frozen.iloc[i], base)[0] == "alert":
                hit = int(i - post.index[0]); break
        first.append((bid, hit))
    print("  首次 alert 距注入的週數：" + ", ".join(f"{b.split('_')[-1]}={h}" for b, h in first))

    print("\n判讀：誤差監控不完美，但它會說 stable 也會說 alert——有鑑別力；"
          "PSI 在本場域兩個都不會說。這不是門檻問題（P4 換三種參考期皆 100% alert）。")


if __name__ == "__main__":
    main()
