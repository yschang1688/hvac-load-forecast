"""P3 結果分析。**宣稱規則寫死在程式裡**，避免事後看 p 值挑形容詞。

規則（與 surface-defect 同一套）：
  p < 0.05          → 可宣稱
  0.05 <= p < 0.10  → 有跡象，不可宣稱
  p >= 0.10         → 落在雜訊帶內，分不出來
另加雜訊帶門檻：差距若小於「配對差異的 2 倍標準誤」，即使 p 過關也標為不可宣稱。
n < 3 一律標「不判定」——Welch 對 n<3 回 nan，而 `nan < 0.10` 是 False，
會靜默落到「落在雜訊帶內」，變成一個讀起來正常卻沒有依據的結論。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def verdict(p: float, diff: float, se2: float) -> str:
    if not np.isfinite(p):
        return "不判定(樣本不足)"
    if abs(diff) < se2:
        return f"雜訊帶內(|差|<2SE={se2:.3f})"
    if p < 0.05:
        return "可宣稱"
    if p < 0.10:
        return "有跡象不可宣稱"
    return "雜訊帶內"


def paired(df: pd.DataFrame, a: str, b: str, metric="skill"):
    """配對比較：同一 (building, fold) 上兩個模型的差異。"""
    key = ["building_id", "fold"]
    x = df[df.model == a].set_index(key)[metric]
    y = df[df.model == b].set_index(key)[metric]
    j = pd.concat([x.rename("a"), y.rename("b")], axis=1).dropna()
    if len(j) < 3:
        return dict(a=a, b=b, n=len(j), mean_a=np.nan, mean_b=np.nan,
                    diff=np.nan, p=np.nan, verdict="不判定(樣本不足)")
    d = j.a - j.b
    t, p = stats.ttest_rel(j.a, j.b)
    se2 = 2 * d.std(ddof=1) / np.sqrt(len(d))
    return dict(a=a, b=b, n=len(j), mean_a=float(j.a.mean()), mean_b=float(j.b.mean()),
                diff=float(d.mean()), p=float(p), verdict=verdict(p, d.mean(), se2))


def main(tag="perfect_forecast"):
    f = ROOT / "reports" / f"p3_results_{tag}.csv"
    if not f.exists():
        print(f"缺 {f}"); return
    df = pd.read_csv(f)
    print(f"=== P3 結果（{tag}）: {df.building_id.nunique()} 棟 × {df.fold.nunique()} folds ===\n")

    agg = df.groupby("model").agg(skill_mean=("skill", "mean"), skill_med=("skill", "median"),
                                  nmae_mean=("nmae", "mean"), n=("skill", "size")).sort_values("skill_mean")
    print("模型總表（skill = MAE ÷ seasonal_naive 的 MAE，<1 才有價值）")
    print(agg.round(4).to_string(), "\n")

    models_ = [m for m in agg.index if m != "seasonal_naive"]
    print("=== 對基準線的配對檢定 ===")
    for m in models_:
        r = paired(df, m, "seasonal_naive")
        print(f"  {m:<18} n={r['n']:>3} skill={r['mean_a']:.3f} vs 1.000  "
              f"diff={r['diff']:+.3f} p={r['p']:.4f}  → {r['verdict']}")

    print("\n=== 策略對照（LightGBM 內部）===")
    for a, b in [("lgbm_direct", "lgbm_recursive")]:
        r = paired(df, a, b)
        print(f"  {a} vs {b}: n={r['n']} diff={r['diff']:+.4f} p={r['p']:.4f} → {r['verdict']}")

    print("\n=== 架構對照 ===")
    for a, b in [("lgbm_direct", "lstm_seq2seq"), ("lgbm_direct", "transformer"),
                 ("lstm_seq2seq", "transformer")]:
        r = paired(df, a, b)
        print(f"  {a} vs {b}: n={r['n']} diff={r['diff']:+.4f} p={r['p']:.4f} → {r['verdict']}")

    # 關機月份的影響
    print("\n=== 分母退化的實證（zero_frac 高的視窗）===")
    hi = df[df.zero_frac > 0.5]
    if len(hi):
        print(f"  zero_frac>0.5 的視窗 {hi[['building_id','fold']].drop_duplicates().shape[0]} 個")
        print(f"  這些視窗的 nMAE 平均 {hi.nmae.mean():.2f}（全體 {df.nmae.mean():.2f}）"
              f"，skill 平均 {hi.skill.mean():.3f}（全體 {df.skill.mean():.3f}）")
        print("  → nMAE 被關機視窗主導，skill 不受影響，這就是改用 skill 的理由")

    # 誤差沿步長
    hf = ROOT / "reports" / f"p3_by_horizon_{tag}.csv"
    if hf.exists():
        h = pd.read_csv(hf)
        piv = h.groupby(["model", "h"]).nmae.mean().unstack()
        print("\n=== 誤差沿預測步長（nMAE，取 h=1/6/12/18/24）===")
        cols = [c for c in (1, 6, 12, 18, 24) if c in piv.columns]
        print(piv[cols].round(3).to_string())
        if {"lgbm_recursive", "lgbm_direct"} <= set(piv.index):
            r1, r24 = piv.loc["lgbm_recursive", 1], piv.loc["lgbm_recursive", 24]
            d1, d24 = piv.loc["lgbm_direct", 1], piv.loc["lgbm_direct", 24]
            print(f"\n  遞迴式 h1→h24 誤差放大 {r24/r1:.2f}×；直接式 {d24/d1:.2f}×")
            print("  （若遞迴放大倍率未明顯高於直接式，就是『誤差累積』這個常識在本資料上不成立，如實報告）")

    out = ROOT / "reports" / f"p3_summary_{tag}.csv"
    agg.to_csv(out)
    print(f"\n落盤 {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "perfect_forecast")
