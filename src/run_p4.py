"""P4 主程式：部署模擬 → 逐週漂移監控 → 自動重訓 → 同協定驗收。

流程（每棟建築各跑一次，設定檔驅動）：
  1. 以 2016 全年訓練初始模型，凍結 PSI 分箱邊界（參考期＝訓練期）
  2. 逐週前推 2017：算該週的 PSI、算該週的實際誤差
  3. PSI 判定為 alert 時觸發重訓（用到當週為止的全部資料）
  4. **重訓後必須以同一套指標在後續週驗收**——重訓不是信仰
  5. 另跑一組「從不重訓」的對照組，兩者相減才是重訓的價值

輸出 reports/p4_drift_{tag}.csv：逐棟逐週的 PSI、判定、是否重訓、誤差。
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import models, dataset, drift  # noqa: E402  （models 必須先載，見其檔頭 OpenMP 註解）

ROOT = Path(__file__).resolve().parents[1]
MONITOR_COLS = ["load_lag_1", "load_lag_24", "load_roll_mean_24",
                "airTemperature_lag1", "dewTemperature_lag1"]


def week_starts(idx: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp):
    return pd.date_range(start, end, freq="7D")


def mae(pred, true):
    m = np.isfinite(pred) & np.isfinite(true)
    return float(np.abs(pred[m] - true[m]).mean()) if m.sum() else np.nan


def run_building(bid, cw, wx, cfg, rows, inject=False):
    H = cfg["modeling"]["horizon"]
    X, F, Y, y = dataset.build_building(bid, cw, wx, cfg)
    if inject:
        y = drift.inject_concept_drift(y, pd.Timestamp(cfg["p4"]["inject_start"]))
        import features
        fcfg = {**cfg["features"], "weather_mode": cfg["modeling"]["weather_mode"]}
        w = dataset.site_weather(wx, bid.split("_")[0])
        X = features.build_design(y, w, fcfg)
        Y = features.build_targets(y, H)
    XF = pd.concat([X, F], axis=1)

    ref_end = pd.Timestamp(cfg["p4"]["ref_end"])
    ref = XF[XF.index <= ref_end]
    edges = {c: drift.psi_bins(ref[c].to_numpy(float)) for c in MONITOR_COLS if c in ref.columns}
    cols = [c for c in MONITOR_COLS if c in edges]

    def fit(upto):
        m = XF.index <= upto
        return models.LGBMDirect(H).fit(XF[m], Y[m])

    model_retrain = fit(ref_end)
    model_frozen = model_retrain                      # 對照組：從不重訓
    last_retrain = ref_end
    weeks = week_starts(XF.index, pd.Timestamp(cfg["p4"]["deploy_start"]),
                        pd.Timestamp(cfg["p4"]["deploy_end"]))

    for ws in weeks:
        we = ws + pd.Timedelta(days=7)
        m = (XF.index >= ws) & (XF.index < we)
        if m.sum() < 24:
            continue
        true = Y[m].to_numpy(float)
        if not np.isfinite(true).any():
            continue
        psis = drift.monitor(ref, XF[m], cols, edges)
        state, max_psi = drift.verdict(psis, cfg["p4"]["psi_warn"], cfg["p4"]["psi_alert"])
        mae_re = mae(model_retrain.predict(XF[m]), true)
        mae_fr = mae(model_frozen.predict(XF[m]), true)
        naive = models.SeasonalNaive().predict_from_series(y, Y[m].index, H)
        mae_nv = mae(naive, true)

        retrained = False
        if state == "alert" and (ws - last_retrain) >= pd.Timedelta(days=cfg["p4"]["min_retrain_gap_days"]):
            model_retrain = fit(ws)                   # 只用到當週為止的資料
            last_retrain = ws
            retrained = True

        rows.append({"building_id": bid, "inject": inject, "week": ws.date().isoformat(),
                     "max_psi": max_psi, "state": state, "retrained": retrained,
                     "mae_retrain": mae_re, "mae_frozen": mae_fr, "mae_naive": mae_nv,
                     "skill_retrain": mae_re / mae_nv if mae_nv else np.nan,
                     "skill_frozen": mae_fr / mae_nv if mae_nv else np.nan,
                     **{f"psi_{c}": psis.get(c, np.nan) for c in cols}})


def main():
    cfg = dataset.load_cfg()
    cw, wx = dataset.load_sources()
    bids = dataset.select_buildings(cfg)[: cfg["p4"]["n_buildings"]]
    rows = []
    for i, b in enumerate(bids, 1):
        t0 = time.time()
        for inj in (False, True):
            try:
                run_building(b, cw, wx, cfg, rows, inject=inj)
            except Exception as e:
                print(f"  !! {b} inject={inj}: {type(e).__name__}: {e}", flush=True)
        print(f"[{i}/{len(bids)}] {b} {time.time()-t0:.0f}s 累計 {len(rows)} 列", flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "reports" / "p4_drift.csv", index=False)
    print("落盤 reports/p4_drift.csv", flush=True)


if __name__ == "__main__":
    main()
