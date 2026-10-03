"""P11 守門測試：機房模型的方向、最佳化器的前瞻行為、安全層、降級、API 契約。

不需要資料庫，也不需要 Bosch 資料：全部用合成輸入。
每條規則都有一個「壞掉就會紅」的反向樣本（探針放在會被執行到的路徑上）。
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mpc  # noqa: E402
import plant_sim as ps  # noqa: E402

VS = ps.PlantConfig(variable_speed=True)
CS = ps.PlantConfig(variable_speed=False)


# ---------------- 機房模型 ----------------
def test_stull_known_value():
    # Stull (2011) 文中例：20 °C、50% → 13.7 °C
    assert abs(float(ps.wet_bulb_stull(20.0, 50.0)) - 13.7) < 0.1


def test_stull_out_of_range_is_nan():
    assert math.isnan(float(ps.wet_bulb_stull(20.0, 2.0)))


def test_raising_chws_lowers_chiller_power_but_raises_flow():
    lo = ps.power(900.0, 15.0, 7.0, 28.0, VS)
    hi = ps.power(900.0, 15.0, 9.0, 28.0, VS)
    assert hi["chiller"] < lo["chiller"]
    assert ps.flow_required(900.0, 9.0, VS) > ps.flow_required(900.0, 7.0, VS)


def test_flow_limit_makes_high_chws_infeasible_near_staging_point():
    # 單台接近加機點（PLR 0.89）：7 °C 可行，10 °C 流量超限
    q = 0.89 * VS.q_unit_kw
    assert bool(ps.power(q, 15.0, 7.0, 28.0, VS)["feasible"])
    assert bool(ps.power(q, 15.0, 10.0, 28.0, VS)["flow_over"])


def test_tower_cannot_beat_wet_bulb():
    assert bool(ps.power(900.0, 25.0, 7.0, 24.0, VS)["tower_short"])


def test_plant_off_draws_no_power():
    assert float(ps.power(0.0, 15.0, 7.0, 28.0, VS)["total"]) == 0.0


# ---------------- 最佳化器 ----------------
def test_solver_respects_slew_from_previous_setpoint():
    chws, cw = mpc.solve((7.0, 30.0), 300.0, 10.0, [300.0] * 7, [10.0] * 7, VS)
    assert abs(chws - 7.0) <= mpc.SLEW_CHWS * mpc.CHWS_STEP + 1e-9
    assert abs(cw - 30.0) <= mpc.SLEW_CW * mpc.CW_STEP + 1e-9


def test_foresight_lowers_chws_before_a_load_ramp():
    """商業邏輯的反向測試：負載即將爬到 10 °C 不可行的區間時，看得到未來的解必須先把供水溫降下來。

    同一個起點（10 °C）、同一個當下負載；差別只有預測。預測壞掉（或最佳化器忽略預測）這條就會紅。
    """
    q_hi = 0.89 * VS.q_unit_kw
    ramp = [400.0, q_hi, q_hi, q_hi, q_hi, q_hi, q_hi]
    flat = [400.0] * 7
    twb = [10.0] * 7
    with_foresight = mpc.solve((10.0, 22.0), 400.0, 10.0, ramp, twb, VS)
    without = mpc.solve((10.0, 22.0), 400.0, 10.0, flat, twb, VS)
    assert without[0] == 10.0
    assert with_foresight[0] < without[0]


def test_optimum_is_not_worse_than_staying_put():
    q, twb = 600.0, 12.0
    chws, cw = mpc.solve((7.0, 30.0), q, twb, [q] * 7, [twb] * 7, VS)
    assert float(ps.power(q, twb, chws, cw, VS)["total"]) <= float(ps.power(q, twb, 7.0, 30.0, VS)["total"]) + 1e-9


def test_constant_speed_chiller_gains_less_from_reset():
    q, twb = 600.0, 12.0
    gain = lambda cfg: float(ps.power(q, twb, 7.0, 28.0, cfg)["chiller"] - ps.power(q, twb, 9.0, 28.0, cfg)["chiller"])  # noqa: E731
    assert gain(CS) < gain(VS)


# ---------------- 安全層 ----------------
@pytest.mark.parametrize("cmd", [(float("nan"), 25.0), (50.0, 25.0), (-3.0, 99.0), (8.0, float("inf"))])
def test_safety_filter_always_returns_within_hard_limits(cmd):
    chws, cw, reasons = mpc.safety_filter(cmd[0], cmd[1], 500.0, 12.0, VS)
    assert mpc.within_hard_limits(chws, cw, VS)
    assert reasons                      # 改過就要說原因


def test_safety_filter_property_random_inputs():
    rng = np.random.default_rng(7)
    for _ in range(2000):
        c1, c2 = rng.uniform(-20, 60, 2)
        q = rng.uniform(0, 3300)
        twb = rng.uniform(-10, 30)
        chws, cw, _ = mpc.safety_filter(c1, c2, q, twb, VS)
        assert mpc.within_hard_limits(chws, cw, VS)
        p = ps.power(q, twb, chws, cw, VS)
        # 過了安全層仍不可行，只允許一種情況：塔扇全開也到不了 32 °C 以下
        assert bool(p["feasible"]) or (bool(p["tower_short"]) and cw == VS.cw_max_c)


def test_safety_filter_lowers_chws_when_flow_would_exceed_limit():
    q = 0.89 * VS.q_unit_kw
    chws, cw, reasons = mpc.safety_filter(10.0, 28.0, q, 12.0, VS)
    assert chws < 10.0 and "flow_limit" in reasons
    assert bool(ps.power(q, 12.0, chws, cw, VS)["feasible"])


def test_safety_filter_leaves_a_good_command_alone():
    chws, cw, reasons = mpc.safety_filter(8.0, 26.0, 500.0, 12.0, VS)
    assert (chws, cw, reasons) == (8.0, 26.0, [])


def test_safety_filter_does_not_depend_on_the_optimizer(monkeypatch):
    """最佳化器回傳垃圾時，control_step 的輸出仍在硬限制內。"""
    monkeypatch.setattr(mpc, "solve", lambda *a, **k: (float("nan"), 1e9))
    d = mpc.control_step((7.0, 30.0), 800.0, 15.0, [800.0] * 7, [15.0] * 7, VS)
    assert mpc.within_hard_limits(d.chws_c, d.cw_c, VS)
    assert "invalid_command" in d.reasons


# ---------------- 降級 ----------------
def test_missing_measurement_falls_back_to_rule_setpoints():
    d = mpc.control_step((9.0, 24.0), float("nan"), 15.0, [500.0] * 7, [15.0] * 7, VS)
    assert d.mode == "fallback" and (d.chws_c, d.cw_c) == (mpc.RULE_CHWS, mpc.RULE_CW)
    assert "missing_measurement" in d.reasons


def test_missing_forecast_is_filled_and_flagged():
    d = mpc.control_step((7.0, 30.0), 500.0, 12.0, [float("nan")] * 7, [12.0] * 7, VS)
    assert d.mode == "mpc" and "forecast_filled" in d.reasons


def test_slew_break_is_reported_when_safety_overrides():
    q = 0.89 * VS.q_unit_kw            # 上一步在 10 °C，當下負載突然跳到 10 °C 不可行的區間
    d = mpc.control_step((10.0, 24.0), q, 12.0, [q] * 7, [12.0] * 7, VS)
    assert d.slew_broken
    assert bool(ps.power(q, 12.0, d.chws_c, d.cw_c, VS)["feasible"])


# ---------------- API 契約 ----------------
@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    import mpc_api
    return TestClient(mpc_api.app)


def test_api_returns_setpoints_within_limits(client):
    r = client.post("/optimize", json={"site_key": "demo", "load_kw": 600, "wet_bulb_c": 12,
                                       "forecast_load_kw": [600] * 7, "forecast_wet_bulb_c": [12] * 7})
    assert r.status_code == 200
    b = r.json()
    assert b["mode"] == "mpc" and mpc.within_hard_limits(b["chws_setpoint_c"], b["cw_setpoint_c"], VS)
    assert b["predicted_total_kw"] > 0


def test_api_rejects_mismatched_forecast_lengths(client):
    r = client.post("/optimize", json={"site_key": "demo", "load_kw": 600, "wet_bulb_c": 12,
                                       "forecast_load_kw": [600] * 7, "forecast_wet_bulb_c": [12] * 3})
    assert r.status_code == 422


def test_api_degrades_when_measurement_missing(client):
    r = client.post("/optimize", json={"site_key": "demo", "load_kw": None, "wet_bulb_c": 12})
    b = r.json()
    assert r.status_code == 200 and b["mode"] == "fallback" and b["predicted_total_kw"] is None
