"""P1 前置：資料剖析。決定哪些建築進入研究樣本，並量出真實的髒資料樣態。

輸出只描述現況，不做任何修補——修補在 quality.py，且必須留 provenance。
"""
from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

def load():
    cw = pd.read_csv(RAW / "chilledwater.csv", parse_dates=["timestamp"])
    meta = pd.read_csv(RAW / "metadata.csv")
    wx = pd.read_csv(RAW / "weather.csv", parse_dates=["timestamp"])
    return cw, meta, wx

def profile_building(s: pd.Series) -> dict:
    n = len(s)
    nn = s.notna().sum()
    out = {"n_rows": n, "n_obs": int(nn), "coverage": nn / n if n else 0.0}
    v = s.dropna()
    if v.empty:
        return {**out, "n_zero": 0, "zero_frac": 0.0, "n_neg": 0,
                "max_gap_h": n, "n_gap_ge6": 0, "flat_max_h": 0, "p99": np.nan, "median": np.nan}
    out["n_zero"] = int((v == 0).sum())
    out["zero_frac"] = float((v == 0).mean())
    out["n_neg"] = int((v < 0).sum())
    out["median"] = float(v.median())
    out["p99"] = float(v.quantile(0.99))
    # 連續缺值長度
    isna = s.isna().to_numpy()
    gaps, run = [], 0
    for x in isna:
        if x: run += 1
        elif run: gaps.append(run); run = 0
    if run: gaps.append(run)
    out["max_gap_h"] = int(max(gaps)) if gaps else 0
    out["n_gap_ge6"] = int(sum(1 for g in gaps if g >= 6))
    # 連續同值（sensor 卡死）——只看非缺值段
    arr = v.to_numpy()
    flat, run = 0, 1
    for i in range(1, len(arr)):
        run = run + 1 if arr[i] == arr[i-1] else 1
        flat = max(flat, run)
    out["flat_max_h"] = int(flat)
    return out

def main():
    cw, meta, wx = load()
    cols = [c for c in cw.columns if c != "timestamp"]
    print(f"chilledwater: {len(cw)} 列 × {len(cols)} 棟建築")
    print(f"時間範圍: {cw.timestamp.min()} → {cw.timestamp.max()}")
    print(f"weather: {len(wx)} 列, {wx.site_id.nunique()} 個 site")
    print(f"metadata: {len(meta)} 棟\n")

    rows = []
    for c in cols:
        r = profile_building(cw[c])
        r["building_id"] = c
        r["site_id"] = c.split("_")[0]
        rows.append(r)
    df = pd.DataFrame(rows).set_index("building_id")

    print("=== 覆蓋率分布 ===")
    print(df.coverage.describe(percentiles=[.1,.25,.5,.75,.9]).round(3).to_string())
    print(f"\n覆蓋率 >=0.95 的建築: {(df.coverage>=.95).sum()}")
    print(f"覆蓋率 >=0.90 的建築: {(df.coverage>=.90).sum()}")
    print(f"\n=== 髒資料樣態（覆蓋率>=0.9 的建築）===")
    d = df[df.coverage >= .90]
    print(f"含負值:            {(d.n_neg>0).sum()} 棟")
    print(f"零值佔比 >20%:     {(d.zero_frac>.2).sum()} 棟")
    print(f"最長 gap >=24h:    {(d.max_gap_h>=24).sum()} 棟")
    print(f"連續同值 >=24h:    {(d.flat_max_h>=24).sum()} 棟")
    print(f"連續同值 >=72h:    {(d.flat_max_h>=72).sum()} 棟")
    print(f"\n=== site 分布（覆蓋率>=0.9）===")
    print(d.groupby("site_id").size().sort_values(ascending=False).to_string())

    outdir = Path(__file__).resolve().parents[1] / "reports"
    outdir.mkdir(exist_ok=True)
    df.to_csv(outdir / "p1_building_profile.csv")
    print(f"\n落盤: {outdir/'p1_building_profile.csv'}")

if __name__ == "__main__":
    main()
