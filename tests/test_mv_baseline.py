"""P9 守門：調整後基準不得依賴報告期能耗；正確做法要回收植入的節能，兩個陷阱要被抓到。

用合成資料（不依賴 Bosch 檔案），CI 上也能跑。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mv_baseline as mb  # noqa: E402

RS = pd.Timestamp("2024-07-01", tz=mb.TZ)


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-01-08", "2024-12-31 23:00", freq="h", tz=mb.TZ)
    doy = idx.dayofyear.to_numpy()
    t = 12 - 12 * np.cos(2 * np.pi * (doy - 15) / 366) + 4 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 1, len(idx))
    occ = ((idx.dayofweek < 5) & (idx.hour >= 8) & (idx.hour < 19)).astype(float)
    # 真實負荷前後小時高度相關（Bosch 冷量正是如此），所以用 AR(1) 而不是白雜訊；
    # 白雜訊下滯後特徵沒有資訊，模型用不到它，陷阱就顯現不出來
    ar = np.zeros(len(idx))
    e = rng.normal(0, 12, len(idx))
    for i in range(1, len(idx)):
        ar[i] = 0.97 * ar[i - 1] + e[i]
    kw = 250 + 150 * occ + 30 * np.maximum(t - 15, 0) + ar
    return pd.DataFrame({"kw": kw, "t": t, "holiday": 0.0}, index=idx)


def total_pct(builder, f, s):
    g = mb.inject_savings(f, RS, s)
    tab = mb.avoided_table(builder(g, RS), g.kw[g.index >= RS])
    return tab.loc["total", "avoided_pct"]


def test_inject_touches_reporting_period_only(frame):
    g = mb.inject_savings(frame, RS, 0.2)
    assert np.allclose(g.kw[g.index < RS], frame.kw[frame.index < RS])
    assert np.allclose(g.kw[g.index >= RS], frame.kw[frame.index >= RS] * 0.8)


def eaten(builder, f, s):
    """被吃掉＝不吸收時應得值 − 算出值；應得值＝1 −（1 − s）×（1 − 安慰劑誤差）。與 run_p9 同一式。"""
    p0 = total_pct(builder, f, 0.0)
    return (1 - (1 - s) * (1 - p0)) - total_pct(builder, f, s)


def test_frozen_baseline_absorbs_nothing(frame):
    """正確做法：安慰劑誤差小，而且植入後的結果**精確**等於不吸收時的應得值（被吃掉＝0）。"""
    assert abs(total_pct(mb.frozen_towt, frame, 0.0)) < 0.03
    assert eaten(mb.frozen_towt, frame, 0.10) == pytest.approx(0.0, abs=1e-9)


def test_guard_passes_frozen_baseline(frame):
    assert mb.depends_on_reporting_energy(mb.frozen_towt, frame, RS) is False


@pytest.mark.parametrize("trap", ["lag_lgbm", "rolling_towt"])
def test_guard_catches_both_traps(frame, trap):
    """探針必須真的會響：兩個陷阱都讓報告期能耗流進基準。"""
    assert mb.depends_on_reporting_energy(mb.BUILDERS[trap], frame, RS) is True


@pytest.mark.parametrize("trap,floor", [("lag_lgbm", 0.08), ("rolling_towt", 0.03)])
def test_traps_eat_savings(frame, trap, floor):
    """植入 15%：滯後能耗吃掉一半以上，滾動重訓也吃掉可觀的一部分（合成資料實測約 10 與 5.5 個百分點）。"""
    assert eaten(mb.BUILDERS[trap], frame, 0.15) > floor


def test_avoided_table_arithmetic():
    idx = pd.date_range("2024-07-01", periods=48, freq="h", tz=mb.TZ)
    tab = mb.avoided_table(pd.Series(100.0, index=idx), pd.Series(90.0, index=idx))
    assert tab.loc["total", "avoided"] == pytest.approx(48 * 10 / 1000)
    assert tab.loc["total", "avoided_pct"] == pytest.approx(0.10)
