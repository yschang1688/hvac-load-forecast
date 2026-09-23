"""P7｜端到端回放：真實 BDG2 建築 × P3 的 LightGBM → 預測落庫 → 回填 → SQL 週 skill → 誤差判定。

補上 P5 報告的誠實邊界「端到端尚未接上 P3 的實際模型」，並在真實資料上比較兩種基準：
- `same_period`：往年同期（John 09-23 裁定）。2016 是訓練期，其誤差若用全年模型算就是 in-sample，
  所以 2016 的預測由**另一半年訓練的模型**產生（H1 模型預測 H2、H2 模型預測 H1），取樣本外誤差。
- `rolling`：最近 8 週。

模型在 2017 全年**凍結**（不重訓），只看監控訊號本身：注入概念漂移（2017-07-01 起，同 P4）前的
alert 是誤報、之後的 alert 是偵測。每天 00:00 發一次 24 步預測。

輸出 reports/p7_replay.csv（逐棟逐週）、reports/p7_replay.txt（摘要）。
"""
from __future__ import annotations
import sys, time, warnings
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import models, dataset, drift, features  # noqa: E402  （models 先載，OpenMP）
import sqlalchemy as sa  # noqa: E402
import db as M, service, schedule  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INJECT = pd.Timestamp("2017-07-01")


def design(bid, cw, wx, cfg, inject):
    H = cfg["modeling"]["horizon"]
    X, F, Y, y = dataset.build_building(bid, cw, wx, cfg)
    if inject:
        y = drift.inject_concept_drift(y, INJECT)
        fcfg = {**cfg["features"], "weather_mode": cfg["modeling"]["weather_mode"]}
        w = dataset.site_weather(wx, bid.split("_")[0])
        X = features.build_design(y, w, fcfg)
        Y = features.build_targets(y, H)
    return pd.concat([X, F], axis=1), Y, y


def fit(XF, Y, lo, hi, H):
    m = (XF.index >= lo) & (XF.index <= hi)
    return models.LGBMDirect(H).fit(XF[m], Y[m])


def daily_origins(lo, hi):
    return pd.date_range(lo, hi, freq="D")


def write(s, site_id, mv_id, model, XF, origins):
    o = origins[origins.isin(XF.index)]
    P = model.predict(XF.loc[o])
    rows = [{"site_id": site_id, "model_version_id": mv_id, "issued_at": t.to_pydatetime(),
             "target_ts": (t + pd.Timedelta(hours=h + 1)).to_pydatetime(), "horizon": h + 1,
             "y_pred": float(P[i, h])}
            for i, t in enumerate(o) for h in range(P.shape[1]) if np.isfinite(P[i, h])]
    for k in range(0, len(rows), 5000):
        s.execute(sa.insert(M.Prediction), rows[k:k + 5000])
    s.commit()
    return len(rows)


def pandas_weekly_skill(s, site_id):
    """獨立於 SQL 的對照計算：同一批列、同一個定義，用 pandas 再算一次。"""
    p = pd.read_sql(sa.select(M.Prediction.issued_at, M.Prediction.target_ts, M.Prediction.y_pred,
                              M.Prediction.y_true).where(M.Prediction.site_id == site_id,
                                                         M.Prediction.y_true.is_not(None)), s.bind)
    o = pd.read_sql(sa.select(M.Observation.ts, M.Observation.chw_load_kw)
                    .where(M.Observation.site_id == site_id), s.bind).set_index("ts").chw_load_kw
    p["naive"] = o.reindex(p.target_ts - pd.Timedelta(hours=168)).to_numpy()
    p = p.dropna(subset=["naive"])
    p["wk"] = p.issued_at.dt.to_period("W-SUN").dt.start_time
    g = p.groupby("wk")
    return (g.apply(lambda d: (d.y_pred - d.y_true).abs().mean()) /
            g.apply(lambda d: (d.naive - d.y_true).abs().mean()))


def run_building(s, bid, cw, wx, cfg, inject, out):
    H = cfg["modeling"]["horizon"]
    XF, Y, y = design(bid, cw, wx, cfg, inject)
    key = f"{bid}{'|inj' if inject else ''}"
    site = M.Site(site_key=key, vendor="vendor_a"); s.add(site); s.flush()
    obs = [{"site_id": site.id, "ts": t.to_pydatetime(), "chw_load_kw": float(v)}
           for t, v in y.dropna().items()]
    for k in range(0, len(obs), 5000):
        s.execute(sa.insert(M.Observation), obs[k:k + 5000])

    t0 = time.time()
    full = fit(XF, Y, pd.Timestamp("2016-01-01"), pd.Timestamp("2016-12-31"), H)
    h1 = fit(XF, Y, pd.Timestamp("2016-01-01"), pd.Timestamp("2016-06-30"), H)
    h2 = fit(XF, Y, pd.Timestamp("2016-07-15"), pd.Timestamp("2016-12-31"), H)
    fit_sec = time.time() - t0
    mv_oof = M.ModelVersion(site_id=site.id, algo="lgbm_direct_oof2016", trained_at=datetime(2016, 1, 1),
                            train_start=datetime(2016, 1, 1), train_end=datetime(2016, 12, 31),
                            retrain_reason="2016 樣本外基準（半年交叉）", is_active=False)
    mv = M.ModelVersion(site_id=site.id, algo="lgbm_direct", trained_at=datetime(2016, 12, 31),
                        train_start=datetime(2016, 1, 1), train_end=datetime(2016, 12, 31),
                        retrain_reason="initial", is_active=True)
    s.add_all([mv_oof, mv]); s.flush()
    # 2016 H1 的預測用 H2 模型（其訓練期 7/15 起，與 H1 的 24 步標的不重疊），反之亦然
    n = write(s, site.id, mv_oof.id, h2, XF, daily_origins("2016-01-11", "2016-06-29"))
    n += write(s, site.id, mv_oof.id, h1, XF, daily_origins("2016-07-15", "2016-12-30"))
    n += write(s, site.id, mv.id, full, XF, daily_origins("2017-01-01", "2017-12-30"))

    weeks = pd.date_range("2017-01-09", "2017-12-18", freq="7D")          # 週一
    for ws in weeks:
        as_of = (ws + pd.Timedelta(days=7)).to_pydatetime()
        for mode_cfg in ("auto", "rolling_only"):
            c = {} if mode_cfg == "auto" else {"same_period_weeks": -1}   # -1 → 同期永遠湊不到 → 滾動
            r = schedule.weekly_error_check(s, key, as_of, c)
            st = r.decision.state if r.decision else r.operating if r.operating != "operating" else "unknown"
            out.append({"building_id": bid, "inject": inject, "week": ws.date(), "policy": mode_cfg,
                        "baseline_mode": r.baseline_mode, "baseline_n": r.baseline_n,
                        "skill": r.skill_now, "baseline": r.baseline,
                        "ratio": r.decision.error_ratio if r.decision else np.nan,
                        "state": st, "operating": r.operating})

    sql = pd.Series({w["week_start"]: w["skill"] for w in schedule.weekly_skill(s, key, datetime(2018, 1, 1))})
    pdv = pandas_weekly_skill(s, site.id)
    j = pd.concat([sql.rename("sql"), pdv.rename("pd")], axis=1).dropna()
    return n, fit_sec, float((j.sql - j.pd).abs().max()), len(j)


def main():
    cfg = dataset.load_cfg()
    cw, wx = dataset.load_sources()
    bids = dataset.select_buildings(cfg)[: cfg["p4"]["n_buildings"]]
    M.Base.metadata.drop_all(service.engine); M.Base.metadata.create_all(service.engine)
    out = []
    with service.SessionLocal() as s:
        for i, b in enumerate(bids, 1):
            for inj in (False, True):
                t0 = time.time()
                n, fs, diff, nw = run_building(s, b, cw, wx, cfg, inj, out)
                print(f"[{i}/{len(bids)}] {b} inject={inj} 預測 {n} 列 fit {fs:.0f}s 共 {time.time()-t0:.0f}s"
                      f"  SQL vs pandas 週 skill 最大差 {diff:.2e}（{nw} 週）", flush=True)
            pd.DataFrame(out).to_csv(ROOT / "reports" / "p7_replay.csv", index=False)
    summarize(pd.DataFrame(out))


def summarize(d: pd.DataFrame):
    d["week"] = pd.to_datetime(d["week"])
    d = d[d.inject]
    print("\n=== 注入組：誤報率（注入前）與偵測率（注入後），模型凍結 ===")
    for pol, g in d.groupby("policy"):
        judged = g[g.state.isin(["stable", "warn", "alert"])]
        pre, post = judged[judged.week < INJECT], judged[judged.week >= INJECT]
        modes = g.baseline_mode.value_counts().to_dict()
        print(f"  {pol:13s} 誤報 {(pre.state=='alert').mean():.0%} (n={len(pre)})  "
              f"偵測 {(post.state=='alert').mean():.0%} (n={len(post)})  "
              f"未判定 {len(g)-len(judged)}  基準來源 {modes}")
    first = []
    for (b, pol), g in d[d.week >= INJECT].groupby(["building_id", "policy"]):
        hit = g[g.state == "alert"].week.min()
        first.append((pol, b.split("_")[-1], None if pd.isna(hit) else (hit - INJECT).days // 7))
    for pol in ("auto", "rolling_only"):
        print(f"  {pol:13s} 首次 alert 距注入週數：" + ", ".join(f"{b}={h}" for p_, b, h in first if p_ == pol))


if __name__ == "__main__":
    main()
