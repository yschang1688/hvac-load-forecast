"""P12 守門測試：調校候選不得破壞節能基準的紀律。全部用合成資料，不需要 Bosch 檔案。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mv_baseline as mb  # noqa: E402
import mv_tuning as mt  # noqa: E402


def _frame(seasonal_level: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """兩年合成資料：時段型態＋溫度效應＋（可選）與溫度無關的冬季水位。"""
    idx = pd.date_range("2023-01-01", "2024-12-31 23:00", freq="h", tz=mb.TZ)
    rng = np.random.default_rng(seed)
    doy = idx.dayofyear.to_numpy()
    t = 12 - 10 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 2, len(idx))
    occ = np.asarray((idx.dayofweek < 5) & (idx.hour >= 7) & (idx.hour < 19), dtype=float)
    winter = np.isin(np.asarray(idx.month), [12, 1, 2]).astype(float)
    kw = 200 + 150 * occ + 12 * np.maximum(t - 10, 0) + seasonal_level * winter + rng.normal(0, 10, len(idx))
    return pd.DataFrame({"kw": kw, "t": t, "holiday": 0.0}, index=idx)


RS = pd.Timestamp("2024-01-01", tz=mb.TZ)


def test_c0_equals_frozen_towt():
    """C0 必須就是現行的 frozen_towt——否則「調校前後」比的不是同一個起點。"""
    f = _frame()
    a = mb.frozen_towt(f, RS)
    b = mt.builder(mt.C0)(f, RS)
    # 兩者只差在正則是否懲罰水位；ridge=1e-3 時逐點差在千分之一以內
    assert np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-3)


@pytest.mark.parametrize("spec", mt.ALL_SPECS, ids=lambda s: s.name)
def test_no_candidate_depends_on_reporting_energy(spec):
    """12 個候選全部要過 P9 的守門：竄改報告期能耗，調整後基準不得改變。
    occ 的高低負載時段若偷用了報告期能耗，這裡會紅。"""
    assert mb.depends_on_reporting_energy(mt.builder(spec), _frame(), RS) is False


def test_guard_catches_occupancy_leak():
    """守門自證：寫一個故意洩漏的版本——高低負載時段用**報告期**的能耗決定。
    P9 的守門必須判它「依賴報告期能耗」；抓不到的話，上一條測試的全綠就沒有意義。"""
    def leaky(frame, rs):
        tr, tg = frame[frame.index < rs], frame[frame.index >= rs]
        occ = mt.occupied_tow(tg)                              # 錯：讀了報告期的 kw
        beta = mt._solve(mt.design(tr, occ), tr.kw.to_numpy(), np.ones(len(tr)), 1e-3)
        return pd.Series(mt.design(tg, occ) @ beta, index=tg.index)

    f = _frame()
    night = (f.index >= RS) & (f.index.hour < 6)
    f.loc[night, "kw"] -= 150.0            # 報告期凌晨偏低；守門把報告期整段放大後，時段排序會變
    g = f.copy()
    g.loc[(g.index >= RS) & (g.index.hour < 6), "kw"] += 5000.0
    assert not np.allclose(leaky(f, RS).to_numpy(), leaky(g, RS).to_numpy())


def test_month_weights_are_cyclic():
    w = mt.month_weights(np.arange(1, 13), 1)
    assert w[0] == 1.0 and w[1] == 0.5 and w[11] == 0.5        # 1 月：當月 1、2 月與 12 月 0.5
    assert w[2:11].sum() == 0.0


def test_segmentation_wins_when_level_is_seasonal_and_not_otherwise():
    """商業邏輯反向測試：冬季有一段與溫度無關的水位時，m3 的樣本外誤差必須明顯低於 none；
    沒有季節水位時，m3 不得更好超過雜訊（否則代表評分邏輯偏袒複雜模型）。"""
    def cv(spec, f):
        base = f[f.index < RS]
        p = mt.oof_predict(spec, base)
        return float(np.sqrt(np.nanmean((base.kw - p) ** 2)) / base.kw.mean())
    seasonal = _frame(seasonal_level=120.0)
    flat = _frame(seasonal_level=0.0)
    none, m3 = mt.Spec("none", "shared", 1.0), mt.Spec("m3", "shared", 1.0)
    assert cv(m3, seasonal) < cv(none, seasonal) * 0.8
    assert cv(m3, flat) > cv(none, flat) * 0.97


def test_injected_savings_are_recovered_out_of_fold():
    f = _frame(seasonal_level=120.0)
    base = f[f.index < RS]
    p = mt.oof_predict(mt.Spec("m3", "shared", 1.0), base)
    rec = (p.sum() - 0.85 * base.kw.sum()) / p.sum()
    assert 0.14 < rec < 0.16


@pytest.mark.parametrize("ridge", mt.RIDGES)
def test_ridge_does_not_shrink_the_level(ridge):
    """正則不得把基準水位往下壓。初版對 168 個時段變數直接做 ridge，ridge=1 時樣本外合計低估 2.7%，
    植入 15% 只算回 12.6%——基準系統性偏低就是憑空吃掉節能。"""
    base = _frame(seasonal_level=120.0)
    base = base[base.index < RS]
    for seg in mt.SEGS:
        p = mt.oof_predict(mt.Spec(seg, "shared", ridge), base)
        assert abs(p.sum() / base.kw.sum() - 1) < 0.005, (seg, ridge)


def test_choose_prefers_simpler_within_tolerance():
    a, b, c = mt.Spec("none", "shared", 1.0), mt.Spec("m3", "shared", 1.0), mt.Spec("m3", "occ", 1.0)
    # none 比最佳差 3 個百分點，出局；m3/shared 在 1 個百分點內且比 m3/occ 簡單
    assert mt.choose({a: 0.320, b: 0.295, c: 0.290}) == b
    # 三者都在 1 個百分點內：選最簡單的
    assert mt.choose({a: 0.298, b: 0.295, c: 0.290}) == a
    # 同結構：取 ridge 較大者
    assert mt.choose({mt.Spec("m3", "shared", 1e-3): 0.290, mt.Spec("m3", "shared", 10.0): 0.291}) == mt.Spec("m3", "shared", 10.0)
