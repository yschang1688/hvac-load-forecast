"""P3｜多步預測的策略 × 架構。

**策略**（怎麼產出 t+1…t+H）：
- recursive  ：單步模型自迴圈，把預測值餵回當輸入。最精簡，但誤差會沿步長累積。
- direct     ：每個 horizon 各訓一個模型。無累積誤差，但要維護 H 個模型（維運成本）。
- multioutput：一次輸出 H 個值。折衷解，也是本專案 NN 架構採用的形式。

**架構**：seasonal-naive（基準線）／LightGBM（統計 ML）／LSTM enc-dec／Transformer。

紀律：所有模型共用同一份特徵、同一組切分、同一套指標；
架構之間**各自調自己的超參**（surface-defect 教訓：共用他人超參會把架構的鍋算錯）。
"""
from __future__ import annotations

# ⚠ 環境陷阱（macOS / Apple Silicon）：lightgbm 與 torch 各自帶一份 OpenMP runtime，
# **torch 先載入時，之後呼叫 LightGBM 的 fit 會直接 segfault（exit 139）**，
# 沒有 Python 例外、沒有錯誤訊息，看起來像程式被殺掉。
# 實測（lightgbm 4.7.0 / torch 2.13.0 / Python 3.14）：
#   import torch → lightgbm.fit          → SIGSEGV
#   import lightgbm → torch → fit        → 正常
#   KMP_DUPLICATE_LIB_OK=TRUE            → 無效
#   OMP_NUM_THREADS=1                    → 可迴避，但犧牲 LightGBM 多執行緒
# 故本檔在最頂端強制先載入 lightgbm。**勿把這行移到 torch 之後，也勿改成延遲載入。**
# 實測補充：光靠載入順序**不夠**——反方向也會炸（先 lightgbm 後建 torch 層同樣 segfault）。
# 唯一在兩個方向都穩的解是把 OpenMP 執行緒數壓到 1，且必須在載入任何 OpenMP 使用者之前設定。
# 代價是 LightGBM 失去多執行緒（本專案單棟訓練約 18 秒，可接受）。
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import lightgbm as _lgb_preload  # noqa: F401  （順序有意義，不可移動）

import numpy as np
import pandas as pd

SEQ_CHANNELS = ["load", "load_lag168", "airTemperature", "dewTemperature",
                "hour_sin", "hour_cos", "is_weekend"]


# ---------- 基準線 ----------
class SeasonalNaive:
    """y_hat(t+h) = y(t+h-168)：上週同一時刻。建築負荷週節律極強，這是誠實的基準線。

    任何模型贏不過它，就沒有存在價值——而它不需要訓練，維運成本為零。
    """
    name = "seasonal_naive"

    def __init__(self, period=168):
        self.period = period

    def fit(self, *a, **k):
        return self

    def predict_from_series(self, y: pd.Series, origins: pd.DatetimeIndex, H: int) -> np.ndarray:
        out = np.full((len(origins), H), np.nan)
        pos = {t: i for i, t in enumerate(y.index)}
        v = y.to_numpy()
        for i, t in enumerate(origins):
            p = pos.get(t)
            if p is None:
                continue
            for h in range(1, H + 1):
                j = p + h - self.period
                if 0 <= j < len(v):
                    out[i, h - 1] = v[j]
        return out


# ---------- LightGBM ----------
def _lgbm(params: dict | None = None):
    lgb = _lgb_preload
    p = dict(objective="l2", n_estimators=250, learning_rate=0.06, num_leaves=31,
             min_child_samples=20, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
             verbose=-1, n_jobs=1)
    p.update(params or {})
    return lgb.LGBMRegressor(**p)


class LGBMDirect:
    """每個 horizon 一個模型。無誤差累積，代價是 H 個模型要一起維護與重訓。"""
    name = "lgbm_direct"

    def __init__(self, H: int, params=None):
        self.H, self.params, self.models = H, params, []

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.models = []
        for h in range(1, self.H + 1):
            m = _lgbm(self.params)
            # ⚠ 只以「標的是否缺值」篩列。**不要再加 X.notna().all(axis=1)**：
            # LightGBM 原生支援缺值特徵，而 BDG2 的 cloudCoverage 在部分 site 缺值達 52%，
            # 多加那個條件會讓可用列從 100% 掉到 29%——被丟掉的不是髒資料，是好資料。
            ok = Y[f"y_h{h}"].notna()
            m.fit(X[ok], Y.loc[ok, f"y_h{h}"])
            self.models.append(m)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models])


class LGBMRecursive:
    """單步模型自迴圈。⚠ 需要把預測值寫回 lag 特徵，因此只能重建「靠負荷 lag」那幾欄。"""
    name = "lgbm_recursive"

    def __init__(self, H: int, lags: list[int], params=None):
        self.H, self.lags, self.params, self.model = H, sorted(lags), params, None

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        ok = Y["y_h1"].notna()          # 同上：特徵缺值交給 LightGBM 自己處理
        self.model = _lgbm(self.params).fit(X[ok], Y.loc[ok, "y_h1"])
        return self

    def predict_recursive(self, X: pd.DataFrame, y: pd.Series, H: int) -> np.ndarray:
        """整批向量化：每個 horizon 對「所有起點」一次預測，而不是逐列逐步呼叫。

        逐列版本在 720 個起點 × 24 步 × 6 folds 下要一萬七千次以上的單列 predict，
        單棟建築就跑掉十幾分鐘；整批版把它壓成 24 次批次 predict。
        """
        cols = list(X.columns)
        lag_cols = {L: cols.index(f"load_lag_{L}") for L in self.lags if f"load_lag_{L}" in cols}
        cur = X.to_numpy(dtype=float).copy()
        n = len(cur)
        out = np.full((n, H), np.nan)
        hist = np.full((n, H), np.nan)          # hist[:, h-1] = 第 h 步的預測值
        # 特徵缺值由 LightGBM 處理；只有整列全缺才沒得預測
        valid = ~np.isnan(cur).all(axis=1)
        for h in range(1, H + 1):
            if not valid.any():
                break
            p = np.full(n, np.nan)
            p[valid] = self.model.predict(cur[valid])
            out[:, h - 1] = p
            hist[:, h - 1] = p
            nxt = cur.copy()
            for L, ci in lag_cols.items():
                if L <= h:
                    nxt[:, ci] = hist[:, h - L]                    # 已由預測值遞補
                else:
                    src = f"load_lag_{L - h}"
                    if src in cols:
                        nxt[:, ci] = X.to_numpy(dtype=float)[:, cols.index(src)]
            cur = nxt
            valid &= ~np.isnan(cur).all(axis=1)
        return out


class LGBMMultiOutput:
    """一次輸出 H 個值（以 H 個獨立學習器實作，但共用一次特徵組裝與一套超參）。"""
    name = "lgbm_multioutput"

    def __init__(self, H: int, params=None):
        self.inner = LGBMDirect(H, params)

    def fit(self, X, Y):
        self.inner.fit(X, Y)
        return self

    def predict(self, X):
        return self.inner.predict(X)


# ---------- 神經網路共用 ----------
def make_sequences(y: pd.Series, wx: pd.DataFrame, origins: pd.DatetimeIndex,
                   L: int, H: int, fut_cols=("airTemperature", "dewTemperature")):
    """組出 (過去序列, 未來外生變數, 標的)。過去只取 t-L+1…t，未來只取天氣與日曆。"""
    idx = y.index
    pos = {t: i for i, t in enumerate(idx)}
    hour = idx.hour.to_numpy(); dow = idx.dayofweek.to_numpy()
    hs, hc = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    we = (dow >= 5).astype(float)
    w = wx.reindex(idx)[list(fut_cols)].astype(float).interpolate(limit=3, limit_area="inside")
    yv = y.to_numpy(dtype=float)
    wv = w.to_numpy(dtype=float)
    # 「上週同時刻」通道：序列窗口為了算力被縮到 5 天，週節律會看不到；
    # 補一條 y[t-168] 讓每個位置都帶著上週同時刻的值，成本是 +1 通道而非 +168 個位置
    # （Transformer 的注意力成本隨位置數平方成長，隨通道數只是線性）。
    ylag = np.full_like(yv, np.nan)
    ylag[168:] = yv[:-168]
    past, fut, tgt, keep = [], [], [], []
    for t in origins:
        p = pos.get(t)
        if p is None or p - L + 1 < 0 or p + H >= len(idx):
            continue
        sl = slice(p - L + 1, p + 1)
        blk = np.column_stack([yv[sl], ylag[sl], wv[sl, 0], wv[sl, 1], hs[sl], hc[sl], we[sl]])
        f = np.column_stack([wv[p + 1:p + 1 + H, 0], wv[p + 1:p + 1 + H, 1],
                             hs[p + 1:p + 1 + H], hc[p + 1:p + 1 + H], we[p + 1:p + 1 + H]])
        t_ = yv[p + 1:p + 1 + H]
        # ⚠ 標的也要檢查。只檢查輸入會讓帶 NaN 的標的流進訓練：
        # L1/L2 loss 遇到 NaN 會回 nan，而 `nan < best` 恆為 False，
        # early stopping 於是永遠不更新最佳權重、val loss 停在 inf，
        # 上層看到 inf 就把整個架構跳過——**沒有例外、沒有警告，模型靜默消失**。
        if np.isnan(blk).any() or np.isnan(f).any() or np.isnan(t_).any():
            continue
        past.append(blk); fut.append(f); tgt.append(t_); keep.append(t)
    if not past:
        return None
    return (np.asarray(past, dtype=np.float32), np.asarray(fut, dtype=np.float32),
            np.asarray(tgt, dtype=np.float32), pd.DatetimeIndex(keep))


# ---------- 神經網路架構 ----------
import torch
import torch.nn as nn


class _Scaler:
    """以訓練段統計量標準化。**必須只用訓練段擬合**，否則測試段的分布資訊會漏進來。"""
    def __init__(self):
        self.mu = self.sd = None

    def fit(self, a: np.ndarray):
        flat = a.reshape(-1, a.shape[-1])
        self.mu = np.nanmean(flat, axis=0)
        self.sd = np.nanstd(flat, axis=0)
        self.sd[self.sd < 1e-6] = 1.0
        return self

    def __call__(self, a):
        return (a - self.mu) / self.sd


class LSTMSeq2Seq(nn.Module):
    """LSTM encoder-decoder：編碼過去 L 小時，解碼時逐步吃未來外生變數輸出 H 步。"""
    name = "lstm_seq2seq"

    def __init__(self, n_past_ch, n_fut_ch, H, hidden=64, layers=1, dropout=0.1):
        super().__init__()
        self.H = H
        self.enc = nn.LSTM(n_past_ch, hidden, layers, batch_first=True)
        self.dec = nn.LSTM(n_fut_ch, hidden, layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, past, fut):
        _, state = self.enc(past)
        out, _ = self.dec(fut, state)          # 以編碼狀態起始，未來外生變數逐步驅動
        return self.head(self.drop(out)).squeeze(-1)


class TimeSeriesTransformer(nn.Module):
    """時序 Transformer：past 與 future 串成一條序列，加上位置編碼與 causal mask。

    causal mask 是這裡的紅線——沒有它，解碼位置會看到更晚的未來位置，
    等於在架構層製造洩漏。P2 的行為檢測抓不到這種（它檢查特徵不檢查注意力）。
    """
    name = "transformer"

    def __init__(self, n_past_ch, n_fut_ch, H, d_model=64, nhead=4, layers=1, dropout=0.1):
        super().__init__()
        self.H = H
        self.pin = nn.Linear(n_past_ch, d_model)
        self.fin = nn.Linear(n_fut_ch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 4096, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout,
                                         batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, past, fut):
        p, f = self.pin(past), self.fin(fut)
        x = torch.cat([p, f], dim=1)
        x = x + self.pos[:, : x.size(1)]
        n = x.size(1)
        mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        x = self.body(x, mask=mask)
        return self.head(x[:, -self.H:]).squeeze(-1)


def train_nn(model, tr, va, epochs=20, lr=1e-3, bs=384, patience=4, device=None, seed=42):
    """訓練迴圈：early stopping 看內層驗證，**不看任何測試資料**。"""
    torch.manual_seed(seed)
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.L1Loss()
    Xp, Xf, Y = [torch.as_tensor(a) for a in tr]
    Vp, Vf, Vy = [torch.as_tensor(a).to(device) for a in va]
    n = len(Xp)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xp[j].to(device), Xf[j].to(device)), Y[j].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Vp, Vf), Vy).item()
        if not np.isfinite(vl):
            raise ValueError(
                "驗證損失非有限值——通常是標的或輸入含 NaN。"
                "不要讓它靜默回 inf：上層會把整個架構跳過而不報錯。")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def predict_nn(model, past, fut, device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval()
    return model(torch.as_tensor(past).to(device), torch.as_tensor(fut).to(device)).cpu().numpy()
