"""P3 主程式：策略 × 架構對照，跨 12 棟建築 × 6 個 rolling-origin fold。

=========================  挑選規則（跑之前寫死）  =========================
1. 主指標＝ nMAE（MAE ÷ 該建築驗證段的平均負荷），使跨建築可比。
   ⚠ **本條在跑完第一棟後被修訂，修訂理由與正當性見下方「指標修訂」段。**
2. 雜訊帶＝同一模型在 (建築 × fold) 上 nMAE 的標準誤 × 2。
   **兩個模型的差距小於雜訊帶即判為分不出來，不得宣稱誰比較好。**
3. 差異顯著性以 Welch t 檢定，p<0.05 可宣稱／0.05–0.10 有跡象／>=0.10 落在雜訊帶內。
4. 架構之間各自調自己的學習率（見 LR_GRID），不共用——共用會把超參的鍋算到架構頭上。
   **lr 只在第一個（時間最早的）fold 上搜尋，選定後凍結給後續 fold 沿用。**
   兩個理由：(a) 算力——每個 fold 都重搜會讓 Transformer 網格跑掉數小時；
   (b) 紀律——每個 fold 都重選超參等於讓模型對每段驗證資料各自最佳化，
   聚合出來的分數會系統性樂觀。凍結後，後續 fold 是真正的前推式驗證。
5. 封存測試段（2017-07 之後）在本階段**完全不開封**。
==========================================================================

指標修訂（2026-08-21，跑完第一棟後）
------------------------------------
**發現**：冰機在冬季會整段關機。第一棟建築的 fold0（2017 年 1 月）有 92% 的小時讀數為 0、
平均負荷僅 4.0（訓練段平均 27.7）。除以這個趨近 0 的分母，任何模型的 nMAE 都會爆掉——
連完全免費、不需訓練的 seasonal naive 都得到 nMAE 2.68。**這是正規化子的結構性缺陷，
不是模型的問題。**

**修訂**：主指標改為 `skill = MAE_model / MAE_seasonal_naive`（同一視窗內），
即「相對於免費基準線的改善倍率」，<1 才代表模型有價值。nMAE 仍照原樣計算並落盤，
不刪除。

**為什麼這不算「看過結果才改指標」**：
判準是**修訂是否可能改變模型排名**。在任一視窗內，seasonal naive 的 MAE 是同一個常數，
所以 skill 只是把該視窗所有模型的分數同乘一個正數——**視窗內排名數學上不可能被翻轉**。
修訂改變的是跨視窗聚合時的權重（不再讓關機月份的極端分母主導平均），
動機是分母退化這個可獨立驗證的事實，不是「哪個模型比較好看」。
對照 surface-defect 的紀律：那裡不可修訂的是**挑選規則**（同分帶怎麼取），
這裡改的是**尺度**，兩者性質不同。原始 nMAE 一併保留供覆核。
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset, models, features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# 序列窗口 120（5 天）而非 168（7 天）：Transformer 的注意力成本隨位置數平方成長，
# 168+24=192 個位置在 M1 的 MPS 上要 10.3 s/epoch，全量網格會跑掉 12 小時。
# 縮到 120+24=144 並把層數減半、批次加大後降到約 1/5。
# 週節律的損失由 make_sequences 的 load_lag168 通道補回（見該函式註解）。
SEQ_LEN = 120
LR_GRID = {"lstm_seq2seq": [3e-3, 1e-3], "transformer": [1e-3, 3e-4]}


def metrics(pred: np.ndarray, true: np.ndarray, denom: float) -> dict:
    m = np.isfinite(pred) & np.isfinite(true)
    if m.sum() == 0:
        return {"mae": np.nan, "rmse": np.nan, "nmae": np.nan, "n": 0}
    e = pred[m] - true[m]
    mae = float(np.abs(e).mean())
    return {"mae": mae, "rmse": float(np.sqrt((e ** 2).mean())),
            "nmae": mae / denom if denom > 0 else np.nan, "n": int(m.sum())}


def per_horizon_mae(pred, true):
    out = []
    for h in range(pred.shape[1]):
        m = np.isfinite(pred[:, h]) & np.isfinite(true[:, h])
        out.append(float(np.abs(pred[m, h] - true[m, h]).mean()) if m.sum() else np.nan)
    return out


def run_building(bid, cw, wx, cfg, rows, hrows):
    frozen_lr: dict[str, float] = {}   # 第一個 fold 選定後凍結（見檔頭規則 4）
    H = cfg["modeling"]["horizon"]
    X, F, Y, y = dataset.build_building(bid, cw, wx, cfg)
    XF = pd.concat([X, F], axis=1)
    w = dataset.site_weather(wx, bid.split("_")[0])
    splits = dataset.rolling_origin_splits(y.index, cfg)

    for sp in splits:
        tr = (y.index <= sp.train_end)
        va = (y.index >= sp.valid_start) & (y.index < sp.valid_end)
        Ytr, Yva = Y[tr], Y[va]
        true = Yva.to_numpy(dtype=float)
        denom = float(np.nanmean(true)) if np.isfinite(true).any() else np.nan
        if not np.isfinite(denom) or denom <= 0:
            continue

        preds = {}
        # --- 基準線 ---
        naive = models.SeasonalNaive().predict_from_series(y, Yva.index, H)
        preds["seasonal_naive"] = naive
        naive_mae = metrics(naive, true, denom)["mae"]

        # --- LightGBM 三策略 ---
        t0 = time.time()
        d = models.LGBMDirect(H).fit(XF[tr], Ytr)
        preds["lgbm_direct"] = d.predict(XF[va])
        r = models.LGBMRecursive(H, cfg["features"]["load_lags"]).fit(XF[tr], Ytr)
        preds["lgbm_recursive"] = r.predict_recursive(XF[va], y, H)
        preds["lgbm_multioutput"] = preds["lgbm_direct"]  # 同一組學習器，策略等價於 direct
        lgbm_s = time.time() - t0

        # --- 神經網路：各自調 lr（內層驗證＝訓練段最後 15%）---
        seq_tr = models.make_sequences(y, w, y.index[tr], SEQ_LEN, H)
        seq_va = models.make_sequences(y, w, Yva.index, SEQ_LEN, H)
        if seq_tr and seq_va and len(seq_tr[0]) > 500:
            Ptr, Ftr, Ttr, _ = seq_tr
            Pva, Fva, Tva, va_idx = seq_va
            cut = int(len(Ptr) * 0.85)
            sp_, sf_ = models._Scaler().fit(Ptr[:cut]), models._Scaler().fit(Ftr[:cut])
            ys_mu, ys_sd = float(Ttr[:cut].mean()), float(Ttr[:cut].std() or 1.0)
            inner_tr = (sp_(Ptr[:cut]), sf_(Ftr[:cut]), (Ttr[:cut] - ys_mu) / ys_sd)
            inner_va = (sp_(Ptr[cut:]), sf_(Ftr[cut:]), (Ttr[cut:] - ys_mu) / ys_sd)
            for arch, ctor in (("lstm_seq2seq", models.LSTMSeq2Seq),
                               ("transformer", models.TimeSeriesTransformer)):
                best_vl, best_pred, best_lr = float("inf"), None, None
                grid = [frozen_lr[arch]] if arch in frozen_lr else LR_GRID[arch]
                for lr in grid:
                    net = ctor(Ptr.shape[-1], Ftr.shape[-1], H)
                    net, vl = models.train_nn(net, inner_tr, inner_va, lr=lr, epochs=20, patience=4)
                    if vl < best_vl:
                        best_vl, best_lr = vl, lr
                        p = models.predict_nn(net, sp_(Pva), sf_(Fva)) * ys_sd + ys_mu
                        full = np.full_like(true, np.nan, dtype=float)
                        loc = {t: i for i, t in enumerate(Yva.index)}
                        for i, t in enumerate(va_idx):
                            if t in loc:
                                full[loc[t]] = p[i]
                        best_pred = full
                if arch not in frozen_lr and best_lr is not None:
                    frozen_lr[arch] = best_lr
                preds[arch] = best_pred

        for name, p in preds.items():
            if p is None:
                continue
            m = metrics(p, true, denom)
            skill = (m["mae"] / naive_mae) if (naive_mae and np.isfinite(naive_mae) and naive_mae > 0) else np.nan
            rows.append({"building_id": bid, "site_id": bid.split("_")[0], "fold": sp.name,
                         "model": name, **m, "skill": skill, "mean_load": denom,
                         "zero_frac": float(np.mean(true[np.isfinite(true)] == 0)),
                         "lgbm_sec": round(lgbm_s, 1)})
            for h, v in enumerate(per_horizon_mae(p, true), start=1):
                hrows.append({"building_id": bid, "fold": sp.name, "model": name,
                              "h": h, "mae": v, "nmae": v / denom if denom > 0 else np.nan})


def main():
    cfg = dataset.load_cfg()
    cw, wx = dataset.load_sources()
    bids = dataset.select_buildings(cfg)
    print(f"建模樣本 {len(bids)} 棟 · horizon {cfg['modeling']['horizon']} · "
          f"weather_mode={cfg['modeling']['weather_mode']}", flush=True)
    rows, hrows = [], []
    for i, b in enumerate(bids, 1):
        t0 = time.time()
        try:
            run_building(b, cw, wx, cfg, rows, hrows)
        except Exception as e:
            print(f"  !! {b}: {type(e).__name__}: {e}", flush=True)
        print(f"[{i}/{len(bids)}] {b}  {time.time()-t0:.0f}s  累計 {len(rows)} 列", flush=True)
        tag_ = cfg["modeling"]["weather_mode"]
        pd.DataFrame(rows).to_csv(ROOT / "reports" / f"p3_results_{tag_}.csv", index=False)
        pd.DataFrame(hrows).to_csv(ROOT / "reports" / f"p3_by_horizon_{tag_}.csv", index=False)
    tag = cfg["modeling"]["weather_mode"]
    pd.DataFrame(rows).to_csv(ROOT / "reports" / f"p3_results_{tag}.csv", index=False)
    pd.DataFrame(hrows).to_csv(ROOT / "reports" / f"p3_by_horizon_{tag}.csv", index=False)
    print(f"落盤 reports/p3_results_{tag}.csv（{len(rows)} 列）", flush=True)


if __name__ == "__main__":
    main()
