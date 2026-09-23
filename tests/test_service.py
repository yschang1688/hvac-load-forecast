"""P5 守門測試：API contract、冪等、閘口攔截。

**跑在真的 PostgreSQL 上**（docker compose up -d），不是 SQLite ——
因為 UPSERT 用的是 PostgreSQL 的 `ON CONFLICT`，換引擎就測不到真正要測的東西。
資料庫不在時整檔 skip，並在報告裡誠實標註沒跑過。
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import sqlalchemy as sa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DB_URL = os.environ.get("HVAC_DB_URL",
                        "postgresql+psycopg2://hvac:hvac_local_dev@localhost:55432/hvac")


def _db_available() -> bool:
    try:
        sa.create_engine(DB_URL).connect().close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="PostgreSQL 未啟動")


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    import db as M
    import service
    M.Base.metadata.drop_all(service.engine)
    M.Base.metadata.create_all(service.engine)
    with TestClient(service.app) as c:
        yield c


def _payload(vendor, n=3, bad=False):
    base = datetime(2017, 8, 1, 0, 0)
    out = []
    for i in range(n):
        ts = (base + timedelta(hours=i)).isoformat()
        if vendor == "vendor_a":
            out.append({"timestamp": ts, "chwLoadKw": -5.0 if bad else 100.0 + i,
                        "oaTempC": 30.0, "dewPointC": 22.0})
        else:
            out.append({"RECORD_TIME": ts, "CHW_LOAD_RT": 28.4 + i,
                        "OA_TEMP_F": 86.0, "DEW_POINT_F": 71.6})
    return out


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["db"] == "postgresql"


def test_two_vendor_dialects_land_on_same_scale(client):
    """兩套方言（kW/°C 與 RT/°F）進來後必須落在同一個尺度上。"""
    a = client.post("/ingest/SITE_A", json={"vendor": "vendor_a", "readings": _payload("vendor_a", 1)})
    b = client.post("/ingest/SITE_B", json={"vendor": "vendor_b", "readings": _payload("vendor_b", 1)})
    assert a.status_code == 200 and b.status_code == 200
    import db as M, service
    with service.SessionLocal() as s:
        vals = s.execute(sa.select(M.QualityEvent.value_raw)
                         .where(M.QualityEvent.action == "flagged")).scalars().all()
    assert len(vals) == 2
    assert abs(vals[0] - vals[1]) < 1.0, f"兩套方言換算後仍差 {abs(vals[0]-vals[1]):.2f} kW"


def test_contract_violation_returns_422_with_reason(client):
    """整批違規要回 422 並帶逐筆理由——不是回 200 然後塞 None 進資料庫。"""
    r = client.post("/ingest/SITE_C", json={"vendor": "vendor_a",
                                            "readings": _payload("vendor_a", 2, bad=True)})
    assert r.status_code == 422
    errs = r.json()["detail"]["errors"]
    assert errs and "field" in errs[0] and "reason" in errs[0]


def test_blocked_rows_leave_an_event(client):
    """被擋下的資料要留紀錄，不是悄悄丟棄——否則事後無從得知漏了什麼。"""
    client.post("/ingest/SITE_D", json={"vendor": "vendor_a",
                                        "readings": _payload("vendor_a", 2, bad=True)})
    import db as M, service
    with service.SessionLocal() as s:
        n = s.scalar(sa.select(sa.func.count()).select_from(M.QualityEvent)
                     .where(M.QualityEvent.action == "blocked"))
    assert n and n > 0


def test_prediction_upsert_is_idempotent(client):
    """同一批預測重放兩次，筆數不得改變（UPSERT 冪等）。"""
    import db as M, service
    client.post("/ingest/SITE_E", json={"vendor": "vendor_a", "readings": _payload("vendor_a", 1)})
    with service.SessionLocal() as s:
        site = s.scalar(sa.select(M.Site).where(M.Site.site_key == "SITE_E"))
        s.add(M.ModelVersion(site_id=site.id, algo="lgbm_direct",
                             train_start=datetime(2016, 1, 1), train_end=datetime(2016, 12, 31),
                             val_skill=0.68, retrain_reason="initial"))
        s.commit()
    body = {"issued_at": datetime(2017, 8, 1, 0, 0).isoformat(), "horizon": 24,
            "y_pred": [float(i) for i in range(24)]}
    r1 = client.post("/predict/SITE_E", json=body)
    r2 = client.post("/predict/SITE_E", json=body)
    assert r1.status_code == 200 and r2.status_code == 200
    with service.SessionLocal() as s:
        n = s.scalar(sa.select(sa.func.count()).select_from(M.Prediction))
    assert n == 24, f"重放後筆數變成 {n}，UPSERT 沒有生效"


def test_predict_without_active_model_is_409(client):
    client.post("/ingest/SITE_F", json={"vendor": "vendor_a", "readings": _payload("vendor_a", 1)})
    r = client.post("/predict/SITE_F", json={"issued_at": datetime(2017, 8, 1).isoformat(),
                                             "horizon": 24, "y_pred": [1.0] * 24})
    assert r.status_code == 409


def test_dashboard_shows_quality_and_business_together(client):
    """儀表板必須同頁顯示業務數字與品質狀態。"""
    client.post("/ingest/SITE_G", json={"vendor": "vendor_a", "readings": _payload("vendor_a", 2)})
    client.post("/ingest/SITE_G", json={"vendor": "vendor_a", "readings": [
        {"timestamp": "2017-08-02T00:00:00", "chwLoadKw": -1.0}]})
    html = client.get("/sites/SITE_G/dashboard").text
    assert "閘口擋下" in html and "最近預測" in html and "模型版本與驗收" in html


def _mk_site(client, key="SITE_S"):
    client.post(f"/ingest/{key}", json={"vendor": "vendor_a", "readings": _payload("vendor_a", 1)})
    return key


def test_weekly_check_respects_cooldown(client):
    """漂移持續存在時不得每週重訓——沒有冷卻期，運維成本與模型抖動都會失控。"""
    import db as M, service, schedule
    key = _mk_site(client, "SITE_CD")
    with service.SessionLocal() as s:
        schedule.register_model_version(s, key, "lgbm_direct", datetime(2016, 1, 1),
                                        datetime(2016, 12, 31), 3.0, 0.7, "initial",
                                        trained_at=datetime(2016, 12, 31))
        hot = {"load_lag_1": 0.9}                    # 遠超 alert 門檻
        d1 = schedule.weekly_retrain_check(s, key, hot, datetime(2017, 1, 8))
        d2 = schedule.weekly_retrain_check(s, key, hot, datetime(2017, 3, 1))
    assert d1.state == "alert" and d1.should_retrain is False and "冷卻期" in d1.reason
    assert d2.should_retrain is True


def test_cooldown_uses_injected_time_not_wall_clock(client):
    """冷卻期必須以模擬時點計算。用牆鐘時間會讓歷史回放算出負間隔，
    冷卻期永遠成立、重訓永遠不觸發，而且全程不報錯。"""
    import service, schedule, db as M
    key = _mk_site(client, "SITE_WC")
    with service.SessionLocal() as s:
        mid = schedule.register_model_version(s, key, "lgbm_direct", datetime(2016, 1, 1),
                                              datetime(2016, 12, 31), 3.0, 0.7, "initial",
                                              trained_at=datetime(2016, 12, 31))
        mv = s.get(M.ModelVersion, mid)
        assert mv.trained_at.year == 2016, f"trained_at 落在 {mv.trained_at}，被牆鐘覆蓋了"


def test_daily_predict_is_idempotent(client):
    import service, schedule, db as M
    key = _mk_site(client, "SITE_DP")
    with service.SessionLocal() as s:
        schedule.register_model_version(s, key, "lgbm_direct", datetime(2016, 1, 1),
                                        datetime(2016, 12, 31), 3.0, 0.7, "initial")
        t = datetime(2017, 8, 1)
        schedule.daily_predict(s, key, t, [1.0] * 24)
        schedule.daily_predict(s, key, t, [2.0] * 24)
        n = s.scalar(sa.select(sa.func.count()).select_from(M.Prediction))
        v = s.scalar(sa.select(M.Prediction.y_pred).where(M.Prediction.horizon == 1))
    assert n == 24, f"重放後 {n} 列"
    assert v == 2.0, "UPSERT 應更新為最新預測值"


def test_register_deactivates_previous_version(client):
    """註冊新版本必須停用舊版本，否則「當前啟用的模型」會有兩個。"""
    import service, schedule, db as M
    key = _mk_site(client, "SITE_MV")
    with service.SessionLocal() as s:
        schedule.register_model_version(s, key, "lgbm_direct", datetime(2016, 1, 1),
                                        datetime(2016, 12, 31), 3.0, 0.70, "initial")
        schedule.register_model_version(s, key, "lgbm_direct", datetime(2016, 1, 1),
                                        datetime(2017, 3, 1), 2.5, 0.62, "psi_alert")
        act = s.execute(sa.select(M.ModelVersion).where(M.ModelVersion.is_active.is_(True))).scalars().all()
    assert len(act) == 1 and act[0].retrain_reason == "psi_alert"


def test_capacity_gate_blocks_overflow(client):
    """有銘牌額定時，超過 1.5 倍的讀數要被擋下（計量器溢位／單位錯誤的典型樣態）。"""
    r = client.post("/ingest/SITE_CAP", json={
        "vendor": "vendor_a", "capacity_kw": 200.0,
        "readings": [{"timestamp": "2017-08-01T00:00:00", "chwLoadKw": 150.0},
                     {"timestamp": "2017-08-01T01:00:00", "chwLoadKw": 1e9}]})
    assert r.status_code == 200
    j = r.json()
    assert j["accepted"] == 1 and j["rejected"] == 1
    assert "銘牌額定" in j["errors"][0]["reason"]


def test_no_capacity_means_no_guess(client):
    """沒有銘牌就不判上限——寧可不判，也不要用資料自身估一個假上限。"""
    r = client.post("/ingest/SITE_NOCAP", json={
        "vendor": "vendor_a",
        "readings": [{"timestamp": "2017-08-01T00:00:00", "chwLoadKw": 1e9}]})
    assert r.status_code == 200 and r.json()["accepted"] == 1


def test_weekly_check_error_signal_overrides_psi(client):
    """P6：傳入誤差訊號時，PSI 再高也不觸發；誤差惡化才觸發（仍受冷卻期約束）。
    這條釘的是「PSI 降級為旁證」——否則 624/624 alert 的 PSI 會把誤差訊號淹掉。"""
    import service, schedule
    key = _mk_site(client, "SITE_ERR")
    with service.SessionLocal() as s:
        schedule.register_model_version(s, key, "lgbm_direct", datetime(2016, 1, 1),
                                        datetime(2016, 12, 31), 3.0, 0.7, "initial",
                                        trained_at=datetime(2016, 12, 31))
        hot = {"load_lag_1": 9.0}                                       # P4 實測量級的 PSI
        ok = schedule.weekly_retrain_check(s, key, hot, datetime(2017, 3, 1),
                                           skill_now=0.62, skill_baseline=0.60)
        bad = schedule.weekly_retrain_check(s, key, hot, datetime(2017, 3, 1),
                                            skill_now=0.95, skill_baseline=0.60)
        cold = schedule.weekly_retrain_check(s, key, hot, datetime(2017, 1, 8),
                                             skill_now=0.95, skill_baseline=0.60)
    assert ok.signal == "error" and ok.state == "stable" and ok.should_retrain is False
    assert ok.max_psi == 9.0                                             # PSI 仍回報，只是不觸發
    assert bad.should_retrain is True and bad.error_ratio > 1.3
    assert cold.should_retrain is False and "冷卻期" in cold.reason
