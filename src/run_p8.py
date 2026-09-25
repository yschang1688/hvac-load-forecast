"""P8｜機房側日前多步冷負荷預測：每天當地 00:00 發一次、96 個 15 分鐘步。

設計與宣稱門檻已事前登記於 `docs/P8_PLAN.md`，本檔照登記執行，不在看到結果後改動。

前置：`src/bosch_plant.py` 的 `build_plant_frame` 與分身點位已輸出到 data/processed/。
    python src/run_p8.py            # 全部 fold × 兩種天氣模式
輸出：reports/p8_daily.csv、reports/p8_by_horizon.csv、reports/p8_summary.txt
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import lightgbm as lgb  # noqa: E402  先於其他 OpenMP 使用者載入（見 models.py 頂端註解）
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bosch_plant import TZ, TRIP_SUPPLY_C  # noqa: E402
from analyze_p3 import verdict  # noqa: E402  與 P3 同一個判定函式，門檻不另寫一份

H = 96                      # 15 分鐘 × 96 ＝ 24 小時
DAY = 96
WEEK = 672
MIN_TRUTH_COVER = 0.95      # 計分日：目標 96 格中至少 95% 有值（事前登記）
MIN_COMMON_STEPS = 0.80     # 各方法共同有效步數低於 80% 則該日不比（基準線歷史缺值時）

# 匈牙利國定假日（Munka Törvénykönyve, 2012. évi I. törvény §102(1)），2024 年：
# 1/1、3/15、耶穌受難日 3/29、復活節週一 4/1、5/1、聖靈降臨節週一 5/20、8/20、10/23、11/1、12/25、12/26。
# 另加 2024 年政府令的兩個橋接休息日 8/19、12/24（未一手核對政府令原文，影響最多兩天）。
HU_HOLIDAYS_2024 = pd.to_datetime([
    "2024-01-01", "2024-03-15", "2024-03-29", "2024-04-01", "2024-05-01", "2024-05-20",
    "2024-08-19", "2024-08-20", "2024-10-23", "2024-11-01", "2024-12-24", "2024-12-25", "2024-12-26",
]).date

PARAM_GRID = [
    dict(n_estimators=250, learning_rate=0.06, num_leaves=31, min_child_samples=20),
    dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20),
    dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20),
    dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50),
]
BASE_PARAMS = dict(objective="l2", subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                   verbose=-1, n_jobs=1, random_state=7)

LAGGED_COLS = ["h", "tod", "dow", "offday", "prev_offday", "y_last", "y_mean_1h", "y_mean_24h",
               "y_max_24h", "y_same_yday", "y_same_lweek", "t_last", "t_mean_24h", "t_max_24h"]
PERFECT_COLS = LAGGED_COLS + ["t_target", "t_mean_day", "t_max_day"]


def load_inputs() -> tuple[pd.Series, pd.Series, pd.Series]:
    tw = pd.read_parquet(ROOT / "data/processed/bosch_twins_15min.parquet")
    pf = pd.read_parquet(ROOT / "data/processed/bosch_plant_15min.parquet")
    y = tw["WC003_load_kw_sensor"].rename("y")
    t = pf["outdoor_c"].reindex(y.index)
    supply = tw["WC003_supply_c"]
    return y, t, supply


def offday(dates) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return ((d.dayofweek >= 5) | np.isin(d.date, HU_HOLIDAYS_2024)).astype(int)


def build_rows(y: pd.Series, t: pd.Series) -> pd.DataFrame:
    """堆疊 (origin × horizon) 列。所有 y_*／t_last 類特徵只取 origin 之前的格點（索引 < p0）。"""
    idx = y.index
    yv, tv = y.to_numpy(float), t.to_numpy(float)
    loc_days = pd.date_range(idx[0].tz_convert(TZ).normalize() + pd.Timedelta(days=8),
                             idx[-1].tz_convert(TZ).normalize() - pd.Timedelta(days=1), freq="D", tz=TZ)
    pos = pd.Series(np.arange(len(idx)), index=idx)
    out = []
    for d0 in loc_days:
        t0 = d0.tz_convert("UTC")
        if t0 not in pos.index:
            continue
        p0 = int(pos[t0])
        if p0 + H > len(idx) or p0 < WEEK:
            continue
        hs = np.arange(1, H + 1)
        j = p0 + hs - 1
        tgt_local = idx[j].tz_convert(TZ)
        with np.errstate(all="ignore"):
            blk = pd.DataFrame({
                "origin": t0, "local_date": d0.date(), "h": hs,
                "tod": tgt_local.hour + tgt_local.minute / 60,
                "dow": tgt_local.dayofweek, "offday": offday(tgt_local.normalize().tz_localize(None)),
                "prev_offday": offday((tgt_local.normalize() - pd.Timedelta(days=1)).tz_localize(None)),
                "y_last": yv[p0 - 1],
                "y_mean_1h": np.nanmean(yv[p0 - 4:p0]),
                "y_mean_24h": np.nanmean(yv[p0 - DAY:p0]),
                "y_max_24h": np.nanmax(yv[p0 - DAY:p0]) if np.isfinite(yv[p0 - DAY:p0]).any() else np.nan,
                "y_same_yday": yv[j - DAY],
                "y_same_lweek": yv[j - WEEK],
                "t_last": tv[p0 - 1],
                "t_mean_24h": np.nanmean(tv[p0 - DAY:p0]),
                "t_max_24h": np.nanmax(tv[p0 - DAY:p0]) if np.isfinite(tv[p0 - DAY:p0]).any() else np.nan,
                # ↓ 只給 perfect_forecast 模式用：實測的未來天氣＝完美預報上界
                "t_target": tv[j],
                "t_mean_day": np.nanmean(tv[p0:p0 + H]),
                "t_max_day": np.nanmax(tv[p0:p0 + H]) if np.isfinite(tv[p0:p0 + H]).any() else np.nan,
                "y": yv[j],
                "target_ts": idx[j],
            })
        out.append(blk)
    return pd.concat(out, ignore_index=True)


def folds(rows: pd.DataFrame):
    """逐月擴張視窗。訓練列的標的時間必須早於測試月第一個 origin（沒有跨界的 24 小時尾巴）。"""
    for m in range(3, 13):
        start = pd.Timestamp(2024, m, 1, tz=TZ).tz_convert("UTC")
        test = rows[(pd.DatetimeIndex(rows.origin).tz_convert(TZ).month == m)]
        train = rows[(rows.target_ts < start) & rows.y.notna()]
        yield m, train, test


def fit_predict(train, test, cols, params):
    m = lgb.LGBMRegressor(**{**BASE_PARAMS, **params})
    m.fit(train[cols], train["y"])
    return m.predict(test[cols])


def score_days(test: pd.DataFrame, preds: dict[str, np.ndarray], trip_days: set) -> pd.DataFrame:
    rec = []
    df = test[["local_date", "h", "y"]].copy()
    for k, v in preds.items():
        df[k] = v
    for day, g in df.groupby("local_date"):
        if g.y.notna().mean() < MIN_TRUTH_COVER:
            continue
        common = g.y.notna()
        for k in preds:
            common &= g[k].notna()
        if common.mean() < MIN_COMMON_STEPS:
            continue
        gg = g[common]
        for k in preds:
            err = (gg[k] - gg.y).abs()
            rec.append(dict(local_date=day, method=k, mae=err.mean(),
                            day_ape=abs(gg[k].sum() - gg.y.sum()) / gg.y.sum(),
                            true_kwh=gg.y.sum() / 4, n_steps=len(gg), trip_day=day in trip_days))
    return pd.DataFrame(rec)


def main():
    y, t, supply = load_inputs()
    rows = build_rows(y, t)
    trip = supply[supply > TRIP_SUPPLY_C]
    trip_days = set(trip.index.tz_convert(TZ).date)
    print(f"rows={len(rows):,}  origins={rows.origin.nunique()}  trip_days={len(trip_days)}")

    # 超參：只看第一個 fold（3 月），lagged_only，選定後凍結
    m3, tr3, te3 = next(folds(rows))
    best, best_mae = None, np.inf
    for p in PARAM_GRID:
        pr = fit_predict(tr3, te3, LAGGED_COLS, p)
        mae = np.nanmean(np.abs(pr - te3.y.to_numpy()))
        print(f"  grid {p} → 3 月 MAE {mae:.1f}")
        if mae < best_mae:
            best, best_mae = p, mae
    print("凍結超參：", best)

    daily, byh = [], []
    for m, tr, te in folds(rows):
        preds = {
            "sn_day": te.y_same_yday.to_numpy(),
            "sn_week": te.y_same_lweek.to_numpy(),
            "lgbm_lagged": fit_predict(tr, te, LAGGED_COLS, best),
            "lgbm_perfect": fit_predict(tr, te, PERFECT_COLS, best),
        }
        d = score_days(te, preds, trip_days)
        d["fold"] = m
        daily.append(d)
        hh = te[["h", "y"]].copy()
        for k, v in preds.items():
            hh[k] = np.abs(v - te.y.to_numpy())
        hh["fold"] = m
        byh.append(hh)
        print(f"fold {m:2d}: train {len(tr):>6,} rows  test days {d.local_date.nunique():>2}  "
              + "  ".join(f"{k}={d[d.method == k].mae.mean():.1f}" for k in preds))
    daily = pd.concat(daily, ignore_index=True)
    daily.to_csv(ROOT / "reports/p8_daily.csv", index=False)
    byh = pd.concat(byh, ignore_index=True)
    (byh.assign(hour_ahead=((byh.h - 1) // 4) + 1)
        .groupby("hour_ahead")[["sn_day", "sn_week", "lgbm_lagged", "lgbm_perfect"]].mean()
        .round(1).to_csv(ROOT / "reports/p8_by_horizon.csv"))
    summarize(daily, best)


def paired_folds(daily: pd.DataFrame, a: str, b: str, metric: str, subset=None) -> dict:
    from scipy import stats
    d = daily if subset is None else daily[subset]
    f = d.pivot_table(index="fold", columns="method", values=metric, aggfunc="mean")
    j = f[[a, b]].dropna()
    if len(j) < 3:
        return dict(a=a, b=b, metric=metric, n=len(j), verdict="不判定(樣本不足)")
    diff = j[a] - j[b]
    _, p = stats.ttest_rel(j[a], j[b])
    se2 = 2 * diff.std(ddof=1) / np.sqrt(len(diff))
    wins = int((diff < 0).sum())
    return dict(a=a, b=b, metric=metric, n=len(j), mean_a=j[a].mean(), mean_b=j[b].mean(),
                diff=diff.mean(), p=p, folds_a_better=f"{wins}/{len(j)}", verdict=verdict(p, diff.mean(), se2))


def summarize(daily: pd.DataFrame, best: dict):
    lines = [f"P8 摘要（凍結超參 {best}）", ""]
    lines.append("逐 fold 平均（日 MAE kW／日總量 APE）：")
    tab = daily.pivot_table(index="fold", columns="method", values=["mae", "day_ape"], aggfunc="mean")
    lines.append(tab.round(3).to_string())
    lines.append("")
    comps = [("lgbm_lagged", "sn_week"), ("lgbm_lagged", "sn_day"), ("sn_day", "sn_week"),
             ("lgbm_perfect", "lgbm_lagged")]
    lines.append("主分析（事前登記：fold 為單位、配對 t、P3 判定）：")
    for metric in ["mae", "day_ape"]:
        for a, b in comps:
            r = paired_folds(daily, a, b, metric)
            lines.append("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    lines.append("")
    lines.append("敏感度分析（排除冰水機跳脫日）：")
    for a, b in comps[:2]:
        r = paired_folds(daily, a, b, "mae", subset=~daily.trip_day)
        lines.append("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    lines.append("")
    w = daily.pivot_table(index="local_date", columns="method", values="mae")
    lines.append(f"輔助描述（逐日勝率，不拿來宣稱）：lgbm_lagged 勝 sn_week {(w.lgbm_lagged < w.sn_week).mean():.0%}、"
                 f"勝 sn_day {(w.lgbm_lagged < w.sn_day).mean():.0%}，計分日 {len(w)}")
    txt = "\n".join(lines)
    (ROOT / "reports/p8_summary.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
