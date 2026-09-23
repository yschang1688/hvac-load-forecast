"""P7｜以視窗層級的關機規則（src/operating_state.py）重新聚合 P3。

P3 的待辦 1：「視窗層級的關機門檻寫死並事前宣告，然後重跑聚合」。
本檔不重訓任何模型，只讀 `reports/p3_results_*.csv`，並從原始負荷重算每個驗證視窗的運轉狀態。

**這仍是敏感度分析，不是事前登錄的檢定**：規則是看過 P3 結果之後寫下的。
所以除了主規則（24 小時、50%），另跑 3×3 門檻網格——結論若在整個網格上都不變，
代表它不依賴門檻的挑選；若會翻，就照實報「取決於門檻」。
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset, operating_state as OS  # noqa: E402
from analyze_p3 import paired  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GRID_HOURS = [12, 24, 48]
GRID_SHARE = [0.3, 0.5, 0.7]


def window_table(cfg) -> pd.DataFrame:
    cw, wx = dataset.load_sources()
    bids = dataset.select_buildings(cfg)[: cfg["modeling"]["n_buildings"]]
    rows = []
    for b in bids:
        y = dataset.build_building(b, cw, wx, cfg)[3]
        for sp in dataset.rolling_origin_splits(y.index, cfg):
            va = (y.index >= sp.valid_start) & (y.index < sp.valid_end)
            yw = y[va]
            r = {"building_id": b, "fold": sp.name}
            for h in GRID_HOURS:
                r[f"share_{h}"] = OS.shutdown_share(yw, h, context=y)
            r["observed"] = float(yw.notna().mean())
            rows.append(r)
    return pd.DataFrame(rows)


def main(tag="perfect_forecast"):
    cfg = dataset.load_cfg()
    oc = {**OS.DEFAULTS, **cfg.get("operating_state", {})}
    res = pd.read_csv(ROOT / "reports" / f"p3_results_{tag}.csv")
    win = window_table(cfg)
    win.to_csv(ROOT / "reports" / "p7_p3_window_state.csv", index=False)

    H, S = oc["min_zero_run_hours"], oc["max_shutdown_share"]
    def excluded(h, s):
        w = win[(win[f"share_{h}"] > s) | (win.observed < oc["min_observed_frac"])]
        return set(zip(w.building_id, w.fold))

    ex = excluded(H, S)
    print(f"=== P3 重新聚合（{tag}）：主規則 零值段 ≥{H}h、佔比 >{S:.0%} → 排除 {len(ex)} 個視窗 ===")
    for k in sorted(ex):
        r = win[(win.building_id == k[0]) & (win.fold == k[1])].iloc[0]
        sk = res[(res.building_id == k[0]) & (res.fold == k[1]) & (res.model == "lgbm_direct")].skill
        print(f"  {k[0]:26s} {k[1]}  關機佔比 {r[f'share_{H}']:.0%}  lgbm_direct skill {sk.iloc[0] if len(sk) else float('nan'):.2f}")

    keep = res[[(b, f) not in ex for b, f in zip(res.building_id, res.fold)]]
    print(f"\n保留 {keep[['building_id','fold']].drop_duplicates().shape[0]} 個視窗")
    for m in ["lgbm_direct", "lgbm_recursive", "lstm_seq2seq", "transformer"]:
        r = paired(keep, m, "seasonal_naive")
        med = keep[keep.model == m].skill.median()
        print(f"  {m:15s} vs 基準線：n={r['n']} mean {r['mean_a']:.3f} median {med:.3f} diff {r['diff']:+.3f} p={r['p']:.4f} → {r['verdict']}")
    for a, b in [("lgbm_direct", "lgbm_recursive"), ("lgbm_direct", "lstm_seq2seq"), ("lgbm_direct", "transformer")]:
        r = paired(keep, a, b)
        print(f"  {a} vs {b}: diff {r['diff']:+.3f} p={r['p']:.4f} → {r['verdict']}")

    print("\n=== 門檻敏感度：lgbm_direct vs 基準線 ===")
    print(f"{'零值段':>6} {'佔比':>5} {'排除':>4} {'skill 平均':>9} {'p':>8}  判定")
    grid = []
    for h in GRID_HOURS:
        for s in GRID_SHARE:
            e = excluded(h, s)
            k = res[[(b, f) not in e for b, f in zip(res.building_id, res.fold)]]
            r = paired(k, "lgbm_direct", "seasonal_naive")
            grid.append(r["verdict"])
            print(f"{h:>5}h {s:>5.0%} {len(e):>4} {r['mean_a']:>9.3f} {r['p']:>8.4f}  {r['verdict']}")
    stable = len(set(v.split("(")[0] for v in grid)) == 1
    print(f"\n結論在 {len(grid)} 組門檻上{'一致' if stable else '不一致——取決於門檻'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "perfect_forecast")
