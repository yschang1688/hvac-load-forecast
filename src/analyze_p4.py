"""P4 結果分析：漂移偵測到了什麼、重訓有沒有用、哪類漂移重訓救不回。

**核心問題不是「PSI 有沒有超標」，是「超標之後重訓有沒有把誤差救回來」。**
重訓是有成本的動作（算力、驗收、模型抖動），沒有驗收就只是儀式。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def claim(p, diff, se2):
    if not np.isfinite(p):
        return "不判定(樣本不足)"
    if abs(diff) < se2:
        return f"雜訊帶內(|差|<2SE={se2:.3f})"
    return "可宣稱" if p < 0.05 else ("有跡象不可宣稱" if p < 0.10 else "雜訊帶內")


def paired(x: pd.Series, y: pd.Series, label: str):
    j = pd.concat([x.rename("a"), y.rename("b")], axis=1).dropna()
    if len(j) < 3:
        print(f"  {label}: n={len(j)} 不判定"); return
    d = j.a - j.b
    t, p = stats.ttest_rel(j.a, j.b)
    se2 = 2 * d.std(ddof=1) / np.sqrt(len(d))
    print(f"  {label}: n={len(j)} 平均差={d.mean():+.4f} p={p:.4f} → {claim(p, d.mean(), se2)}")


def main():
    f = ROOT / "reports" / "p4_drift.csv"
    if not f.exists():
        print("缺 p4_drift.csv"); return
    d = pd.read_csv(f, parse_dates=["week"])
    print(f"=== P4：{d.building_id.nunique()} 棟 × {d.week.nunique()} 週 × 2 組（未注入/注入）===\n")

    for inj, sub in d.groupby("inject"):
        tag = "注入概念漂移" if inj else "未注入（自然漂移）"
        print(f"--- {tag} ---")
        print("  PSI 判定分布:", dict(sub.state.value_counts()))
        print(f"  觸發重訓次數: {int(sub.retrained.sum())}（{sub.building_id.nunique()} 棟）")
        ok = sub.dropna(subset=["skill_retrain", "skill_frozen"])
        print(f"  skill 平均：重訓組 {ok.skill_retrain.mean():.3f} / 凍結組 {ok.skill_frozen.mean():.3f}"
              f"（中位數 {ok.skill_retrain.median():.3f} / {ok.skill_frozen.median():.3f}）")
        paired(ok.set_index(["building_id", "week"]).skill_retrain,
               ok.set_index(["building_id", "week"]).skill_frozen, "重訓 vs 凍結（全期）")
        print()

    print("=== 注入後的時期（漂移已發生）===")
    inj = d[d.inject]
    if len(inj):
        start = pd.Timestamp("2017-07-01")
        post = inj[inj.week >= start].dropna(subset=["skill_retrain", "skill_frozen"])
        pre = inj[inj.week < start].dropna(subset=["skill_retrain", "skill_frozen"])
        for nm, s in (("注入前", pre), ("注入後", post)):
            if len(s):
                print(f"  {nm}: 重訓 {s.skill_retrain.mean():.3f} / 凍結 {s.skill_frozen.mean():.3f}"
                      f"  PSI alert 週數 {int((s.state=='alert').sum())}/{len(s)}")
        if len(post):
            paired(post.set_index(["building_id", "week"]).skill_retrain,
                   post.set_index(["building_id", "week"]).skill_frozen, "注入後 重訓 vs 凍結")

    print("\n=== 概念漂移 PSI 抓得到嗎？（注入 vs 未注入的 PSI 對照）===")
    piv = d.pivot_table(index=["building_id", "week"], columns="inject", values="max_psi")
    piv.columns = ["no_inject", "inject"]
    piv = piv.dropna()
    post = piv[piv.index.get_level_values("week") >= pd.Timestamp("2017-07-01")]
    if len(post):
        print(f"  注入後：未注入 PSI 中位數 {post.no_inject.median():.3f} / "
              f"注入 PSI 中位數 {post.inject.median():.3f}")
        rise = (post.inject > post.no_inject).mean()
        print(f"  注入使 PSI 上升的週佔比 {rise:.1%}")
        print("  → PSI 看的是**輸入分布**。概念漂移改的是輸入到輸出的映射，"
              "若輸入分布沒動，PSI 就抓不到——這正是演練要證明的事。")

    print("\n=== 誤報檢查：PSI 說 alert 但誤差沒惡化的週 ===")
    ok = d.dropna(subset=["skill_frozen"])
    al = ok[ok.state == "alert"]; st = ok[ok.state == "stable"]
    if len(al) and len(st):
        print(f"  alert 週的凍結組 skill 中位數 {al.skill_frozen.median():.3f}"
              f" vs stable 週 {st.skill_frozen.median():.3f}")
        print("  → 兩者若接近，代表 PSI 的 alert 與實際誤差惡化沒有對應，"
              "這種告警上線後會被維運忽略。")
    d.to_csv(ROOT / "reports" / "p4_drift.csv", index=False)


if __name__ == "__main__":
    main()
