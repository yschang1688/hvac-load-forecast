"""P13｜日曆特徵補測：A 拿掉時段特徵會差多少；B 學校建築的寒假旗標。設計見 docs/P13_PLAN.md（跑之前提交）。"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import models, dataset
import numpy as np, pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
HS = [1, 6, 12, 18, 24]
GROUPS = {"-hour": ["hour_sin", "hour_cos"], "-dow": ["dow", "is_weekend"], "-month": ["month_sin", "month_cos"]}
GROUPS["-calendar"] = sum(GROUPS.values(), [])
FED = set(USFederalHolidayCalendar().holidays("2015-12-01", "2018-01-31").date)


def is_winter_break(d) -> bool:
    """通用近似（寫死在 docs/P13_PLAN.md）：12/20 到隔年 1/8。"""
    return (d.month == 12 and d.day >= 20) or (d.month == 1 and d.day <= 8)


def flag_block(idx, fn, name):
    f = pd.Series([int(fn(d)) for d in idx.date], index=idx)
    out = pd.DataFrame({f"{name}_h{h}": f.shift(-h) for h in HS}, index=idx)
    out[f"is_{name}"] = f
    return out


def education_buildings(cfg, n=12):
    v = pd.read_csv(ROOT / "reports" / "p1_quality_verdicts.csv")
    m = cfg["modeling"]
    ok = v[(v.verdict != "Reject") & (v.coverage >= m["min_coverage"]) & (v.nonzero_frac >= m.get("min_nonzero_frac", 0.5))
           & (v.building_id.str.split("_").str[1] == "education")].copy()
    ok = ok.sort_values(["site_id", "error_frac", "coverage"], ascending=[True, True, False])
    return ok.groupby("site_id").head(m["per_site"]).head(n).building_id.tolist()


def run(bids, make_variants, cw, wx, cfg, count_col=None):
    rows = []
    for b in bids:
        X, F, Y, y = dataset.build_building(b, cw, wx, cfg)
        variants, counter = make_variants(pd.concat([X, F], axis=1), y.index)
        for sp in dataset.rolling_origin_splits(y.index, cfg):
            tr = (y.index <= sp.train_end)
            va = (y.index >= sp.valid_start) & (y.index < sp.valid_end)
            nv = models.SeasonalNaive().predict_from_series(y, y.index[va], cfg["modeling"]["horizon"])
            for h in HS:
                t = Y.loc[va, f"y_h{h}"].to_numpy(float)
                mnv = np.nanmean(np.abs(nv[:, h - 1] - t))
                if not np.isfinite(mnv) or mnv == 0:
                    continue
                r = {"b": b, "fold": sp.name, "h": h, "zero_frac": float((y[va] == 0).mean())}
                ok = Y.loc[tr, f"y_h{h}"].notna()
                for k, V in variants.items():
                    m = models._lgbm().fit(V[tr][ok], Y.loc[tr, f"y_h{h}"][ok])
                    r[k] = np.nanmean(np.abs(m.predict(V[va]) - t)) / mnv
                if counter is not None:
                    r[count_col] = int((counter.loc[va, f"{count_col}_h{h}"] == 1).sum())
                rows.append(r)
        print(b, len(rows), flush=True)
    return pd.DataFrame(rows)


def summarize(d, keys, title, out):
    for nm, s in ((f"{title}｜全部", d), (f"{title}｜排除關機視窗 zero_frac>0.5", d[d.zero_frac <= 0.5])):
        out.append(f"\n=== {nm} n={len(s)} ===")
        for k in keys:
            diff = s[k] - s["base"]
            _, p = stats.ttest_rel(s[k], s["base"])
            se2 = 2 * diff.std(ddof=1) / np.sqrt(len(diff))
            verdict = "雜訊帶內" if abs(diff.mean()) < se2 or p >= 0.10 else ("可宣稱" if p < 0.05 else "有跡象不可宣稱")
            per_b = s.assign(d=diff).groupby("b").d.mean()
            out.append(f"{k:10s} skill 中位數 {s[k].median():.3f}（base {s['base'].median():.3f}） 平均差 {diff.mean():+.4f} "
                       f"2SE={se2:.4f} p={p:.4f} → {verdict}｜逐棟：差>0 {int((per_b > 0).sum())} 棟、差<0 {int((per_b < 0).sum())} 棟")
        out.append("逐步長平均差：")
        out.append(s.assign(**{k: s[k] - s["base"] for k in keys}).groupby("h")[keys].mean().round(4).to_string())


if __name__ == "__main__":
    cfg = dataset.load_cfg(); cw, wx = dataset.load_sources()
    out = [f"P13 摘要（weather_mode={cfg['modeling']['weather_mode']}；差＝變體 − base）"]

    def drop_variants(base, idx):
        missing = [c for c in GROUPS["-calendar"] if c not in base.columns]
        assert not missing, f"基礎特徵缺欄位：{missing}"
        return {"base": base, **{k: base.drop(columns=cols) for k, cols in GROUPS.items()}}, None

    a = run(dataset.select_buildings(cfg)[:12], drop_variants, cw, wx, cfg)
    a.to_csv(ROOT / "reports" / "p13_time_ablation.csv", index=False)
    out.append("\n實驗 A：差為正＝拿掉後變差＝該特徵有增益")
    summarize(a, list(GROUPS), "A 拿掉時段特徵", out)

    def add_variants(base, idx):
        fed, brk = flag_block(idx, lambda d: d in FED, "fed"), flag_block(idx, is_winter_break, "break")
        return {"base": base, "+fed": pd.concat([base, fed], axis=1), "+break": pd.concat([base, brk], axis=1),
                "+both": pd.concat([base, fed, brk], axis=1)}, brk

    edu = education_buildings(cfg)
    out.append(f"\n實驗 B 樣本（{len(edu)} 棟）：{', '.join(edu)}")
    b = run(edu, add_variants, cw, wx, cfg, count_col="break")
    b.to_csv(ROOT / "reports" / "p13_school_break.csv", index=False)
    out.append("實驗 B：差為負＝加入後改善")
    summarize(b, ["+fed", "+break", "+both"], "B 學校建築加旗標", out)
    out.append(f"\n驗證期含寒假時點的列：{int((b['break'] > 0).sum())} / {len(b)}（依 fold：{b[b['break'] > 0].fold.value_counts().to_dict()}）")
    hb = b[b["break"] > 0]
    if len(hb) >= 3:
        diff = hb["+break"] - hb["base"]; _, p = stats.ttest_rel(hb["+break"], hb["base"])
        out.append(f"只看含寒假的列（描述性，n={len(hb)}）：+break 平均差 {diff.mean():+.4f} p={p:.4f}")
    (ROOT / "reports" / "p13_summary.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
