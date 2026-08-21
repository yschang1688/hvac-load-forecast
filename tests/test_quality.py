"""品質閘的自證測試：植入缺陷 → 對應規則必須響。

**為什麼需要這種測試**：一個從不作響的閘門，和一個壞掉的閘門，在乾淨資料上看起來一模一樣。
真實資料全過不是閘門有效的證據，逐型植入才是。

**探針要真的壞**（salary-mcp-agent 的教訓：探針放在永不執行的位置＝沒植入）——
本檔每個植入案例都先斷言「乾淨版本不觸發」，再斷言「植入版本觸發」，
兩個斷言都過才代表規則有鑑別力而不是恆真。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import quality  # noqa: E402


def clean_series(n=24 * 30, seed=42):
    """合成一段乾淨的冰水負荷：日內週期＋週間差異＋小雜訊，全部為正。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-06-01", periods=n, freq="h")
    hour = idx.hour.to_numpy()
    daily = 60 + 40 * np.sin((hour - 6) / 24 * 2 * np.pi)
    weekend = np.where(idx.dayofweek.to_numpy() >= 5, 0.6, 1.0)
    v = daily * weekend + rng.normal(0, 2.0, n)
    return pd.Series(np.clip(v, 1.0, None), index=idx)


def fired(s, code, cfg=None):
    flags, _ = quality.evaluate(s, cfg or {})
    return bool(flags[code].any())


def test_clean_series_is_pass():
    """基準線：乾淨資料不該觸發任何 ERROR 級規則，否則後面每個植入測試都是恆真。"""
    _, summary = quality.evaluate(clean_series())
    assert summary["n_error"] == 0, summary
    assert summary["verdict"] in ("Pass", "Conditional")


@pytest.mark.parametrize("code,inject", [
    ("R001", lambda s: s.copy().pipe(lambda x: x.mask(x.index == x.index[100], -5.0))),
    ("R002", lambda s: s.copy().pipe(lambda x: x.mask(x.index == x.index[200], x.max() * 5000))),
    ("R003", lambda s: s.copy().pipe(lambda x: x.mask(x.index == x.index[300], x.median() * 60))),
    ("R004", lambda s: _flat(s, at=400, hours=48, value=77.0)),
    ("R005", lambda s: _gap(s, at=500, hours=3)),
    ("R006", lambda s: _gap(s, at=600, hours=12)),
])
def test_injected_defect_is_caught(code, inject):
    base = clean_series()
    assert not fired(base, code), f"{code} 在乾淨資料上就響了，規則沒有鑑別力"
    assert fired(inject(base), code), f"{code} 沒抓到植入的缺陷"


def _flat(s, at, hours, value):
    out = s.copy()
    out.iloc[at:at + hours] = value
    return out


def _gap(s, at, hours):
    out = s.copy()
    out.iloc[at:at + hours] = np.nan
    return out


def test_zero_flat_run_is_not_flagged():
    """領域規則的核心：冰機關機時讀數為 0，長時間零值平坦是正常運轉樣態不是故障。

    這條測試擋的是「把 R004 寫成不分零值」的退化——實資料上該退化會製造約 70% 誤報。
    """
    s = clean_series()
    s.iloc[400:400 + 200] = 0.0          # 關機 200 小時
    assert not fired(s, "R004"), "零值平坦段被誤判為 sensor 卡死"


def test_raw_values_are_never_mutated():
    """原始值不刪不改：clean() 只能新增欄位，value_raw 必須與輸入逐格相同（含 NaN 位置）。"""
    s = _gap(_flat(clean_series(), at=100, hours=48, value=77.0), at=300, hours=10)
    flags, _ = quality.evaluate(s)
    out = quality.clean(s, flags)
    pd.testing.assert_series_equal(out["value_raw"], s, check_names=False)


def test_imputation_is_recorded():
    """補過的值一定要留 provenance，否則髒資料寫進庫就再也看不出來。"""
    s = _gap(clean_series(), at=300, hours=3)
    flags, _ = quality.evaluate(s)
    out = quality.clean(s, flags)
    filled = out["value_clean"].notna() & out["value_raw"].isna()
    assert filled.any(), "短缺口未被插補"
    assert (out.loc[filled, "impute_method"] == "linear").all()
    assert out.loc[filled, "imputed"].all()


def test_long_gap_is_not_interpolated():
    """長缺口不得插補——插補會捏造出不存在的日內尖峰。"""
    s = _gap(clean_series(), at=300, hours=12)
    flags, _ = quality.evaluate(s)
    out = quality.clean(s, flags)
    assert out["value_clean"].iloc[300:312].isna().all(), "長缺口被插補了"


def test_streaming_safe_rules_do_not_look_ahead():
    """streaming_safe 的規則只能看當下與過去：把序列尾端整段改掉，
    前段的判定必須逐格不變。會變的規則就是偷看了未來。"""
    s = clean_series()
    tampered = s.copy()
    tampered.iloc[-200:] = tampered.iloc[-200:] * 100
    cut = len(s) - 200
    f1, _ = quality.evaluate(s)
    f2, _ = quality.evaluate(tampered)
    for r in quality.RULES:
        if not r.streaming_safe:
            continue
        pd.testing.assert_series_equal(
            f1[r.code].iloc[:cut], f2[r.code].iloc[:cut],
            check_names=False, obj=f"{r.code} 的過去判定被未來資料改變了")


def test_zero_inflated_series_does_not_degenerate_ceiling():
    """零膨脹退化守衛（R002）：冰機大部分時間關機時，含零計算的 IQR 會塌成 0，
    使「超出物理上限」退化成「任何非零讀數都違規」。本測試把該退化釘死。

    這是本專案第三次踩到同一個根因（R002 上限、R003 尺度、R004 平坦段），
    故三條規則都改為在非零子集上估計尺度。
    """
    base = clean_series()
    s = base.copy()
    off = np.zeros(len(s), dtype=bool)
    off[: int(len(s) * 0.8)] = True          # 80% 時間關機
    s[off] = 0.0
    flags, _ = quality.evaluate(s)
    n_nonzero = int((s > 0).sum())
    assert flags["R002"].sum() < n_nonzero * 0.5, (
        f"零膨脹下 R002 把 {int(flags['R002'].sum())}/{n_nonzero} 個非零讀數判成超上限，尺度已退化")


def test_spike_scale_survives_flat_overnight():
    """R003 尺度守衛：夜間長時間平坦會讓『過去 diff 中位數』趨近 0，
    若以它當尺度，白天正常爬升就會觸發。改以運轉水準為尺度後不該如此。"""
    s = clean_series()
    for d in range(30):                        # 每天 0–6 時壓成同一個低值
        s.iloc[d * 24: d * 24 + 7] = 5.0
    flags, _ = quality.evaluate(s)
    rate = flags["R003"].mean()
    assert rate < 0.05, f"R003 命中率 {rate:.1%}，夜間平坦把門檻壓垮了"
