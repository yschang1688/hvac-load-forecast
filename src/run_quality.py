"""P1 主程式：一套 pipeline、一份設定檔，跑完全部案場。

公版化的驗收條件：新增一棟建築只改 config/pipeline.json，**不改本檔任何一行**。
"""
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config" / "pipeline.json").read_text(encoding="utf-8"))


def main():
    raw = ROOT / "data" / "raw"
    cw = pd.read_csv(raw / CFG["data"]["meter_file"], parse_dates=["timestamp"]).set_index("timestamp")
    sel, qcfg = CFG["selection"], CFG["quality"]

    rows = []
    for bid in cw.columns:
        s = cw[bid]
        cov = s.notna().mean()
        nz = (s.dropna() != 0).mean() if s.notna().any() else 0.0
        if cov < sel["min_coverage"] or nz < sel["min_nonzero_frac"]:
            continue
        _, summary = quality.evaluate(s, qcfg)
        summary.update(building_id=bid, site_id=bid.split("_")[0],
                       coverage=round(float(cov), 4), nonzero_frac=round(float(nz), 4))
        rows.append(summary)

    df = pd.DataFrame(rows).set_index("building_id")
    out = ROOT / "reports" / "p1_quality_verdicts.csv"
    df.to_csv(out)

    print(f"入選案場: {len(df)} 棟（覆蓋率>={sel['min_coverage']}、非零佔比>={sel['min_nonzero_frac']}）")
    print(f"\n=== 三級判定分布 ===")
    print(df.verdict.value_counts().to_string())
    print(f"\n=== 各規則觸發的案場數 ===")
    for r in quality.RULES:
        n_bld = int((df[r.code] > 0).sum())
        n_pts = int(df[r.code].sum())
        print(f"  {r.code} {r.severity:<5} {r.name:<24} {n_bld:>3} 棟 / {n_pts:>7} 點")
    print(f"\n=== 每個 site 的判定 ===")
    print(pd.crosstab(df.site_id, df.verdict).to_string())
    print(f"\n落盤: {out}")


if __name__ == "__main__":
    main()
