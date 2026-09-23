"""P6｜三個外部建議特徵的消融：熱慣性（過去氣溫 3／6 小時滾動與指數衰減）、國定假日。結果見 docs/DECISION_P6_FEATURES.md。"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import models, dataset
import numpy as np, pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy import stats
cfg = dataset.load_cfg(); cw, wx = dataset.load_sources()
bids = dataset.select_buildings(cfg)[:12]
hol = set(USFederalHolidayCalendar().holidays("2015-12-01", "2018-01-31").date)
HS = [1, 6, 12, 18, 24]
rows = []
for b in bids:
    X, F, Y, y = dataset.build_building(b, cw, wx, cfg)
    w = dataset.site_weather(wx, b.split("_")[0]).reindex(y.index)["airTemperature"].interpolate(limit=3, limit_area="inside")
    extra = pd.DataFrame({
        "airTemp_roll3": w.shift(1).rolling(3, min_periods=2).mean(),
        "airTemp_roll6": w.shift(1).rolling(6, min_periods=3).mean(),
        "airTemp_ewm6": w.shift(1).ewm(halflife=6).mean(),
    }, index=y.index)
    # 假日：預測目標時點 t+h 是否假日，才是有用的訊號；t 本身是否假日只是近似
    hday = pd.Series([d in hol for d in y.index.date], index=y.index).astype(int)
    Hf = pd.DataFrame({f"hol_h{h}": hday.shift(-h) for h in HS}, index=y.index)  # 日曆可事前得知，非洩漏
    Hf["is_holiday"] = hday
    base = pd.concat([X, F], axis=1)
    variants = {"base": base, "+thermal": pd.concat([base, extra], axis=1),
                "+holiday": pd.concat([base, Hf], axis=1), "+both": pd.concat([base, extra, Hf], axis=1)}
    for sp in dataset.rolling_origin_splits(y.index, cfg):
        tr = (y.index <= sp.train_end)
        va = (y.index >= sp.valid_start) & (y.index < sp.valid_end)
        nv = models.SeasonalNaive().predict_from_series(y, y.index[va], cfg["modeling"]["horizon"])
        for h in HS:
            t = Y.loc[va, f"y_h{h}"].to_numpy(float)
            mnv = np.nanmean(np.abs(nv[:, h-1] - t))
            if not np.isfinite(mnv) or mnv == 0: continue
            r = {"b": b, "fold": sp.name, "h": h,
                 "zero_frac": float((y[va] == 0).mean())}
            for k, V in variants.items():
                ok = Y.loc[tr, f"y_h{h}"].notna()
                m = models._lgbm().fit(V[tr][ok], Y.loc[tr, f"y_h{h}"][ok])
                p = m.predict(V[va])
                r[k] = np.nanmean(np.abs(p - t)) / mnv
                if k == "+holiday":
                    hm = (Hf.loc[va, f"hol_h{h}"] == 1).to_numpy()
                    r["hol_n"] = int(hm.sum())
            rows.append(r)
    print(b, len(rows), flush=True)
d = pd.DataFrame(rows)
d.to_csv("" + str(__import__("pathlib").Path(__file__).resolve().parents[1] / "reports") + "/p6_feature_ablation.csv", index=False)
keep = d[d.zero_frac <= 0.5]
for nm, s in (("全部", d), ("排除關機視窗 zero_frac>0.5", keep)):
    print(f"\n=== {nm} n={len(s)} ===")
    for k in ["+thermal", "+holiday", "+both"]:
        diff = s[k] - s["base"]; t, p = stats.ttest_rel(s[k], s["base"])
        se2 = 2 * diff.std(ddof=1) / np.sqrt(len(diff))
        verdict = "雜訊帶內" if abs(diff.mean()) < se2 or p >= 0.10 else ("可宣稱" if p < 0.05 else "有跡象")
        print(f"{k:9s} skill {s[k].median():.3f} vs base {s['base'].median():.3f}  平均差 {diff.mean():+.4f} p={p:.3f} → {verdict}")
    print("逐 horizon 中位數差（+thermal / +holiday）:")
    print(s.assign(dt=s["+thermal"]-s.base, dh=s["+holiday"]-s.base).groupby("h")[["dt","dh"]].median().round(4).T.to_string())
print("\n驗證期含假日的列:", int((d.hol_n > 0).sum()), "/", len(d))
