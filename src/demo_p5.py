"""P5 端到端實跑：兩套廠牌方言 → 閘口 → 落庫 → 預測 → 儀表板。產出存證。"""
import sys
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi.testclient import TestClient
import sqlalchemy as sa
import db as M, service, schedule

M.Base.metadata.drop_all(service.engine); M.Base.metadata.create_all(service.engine)
c = TestClient(service.app)
base = datetime(2017, 8, 1)

a = [{"timestamp": (base+timedelta(hours=i)).isoformat(), "chwLoadKw": 100.0+i,
      "oaTempC": 30.0+i*0.1, "dewPointC": 22.0} for i in range(48)]
b = [{"RECORD_TIME": (base+timedelta(hours=i)).isoformat(), "CHW_LOAD_RT": 28.4+i*0.28,
      "OA_TEMP_F": 86.0+i*0.18, "DEW_POINT_F": 71.6} for i in range(48)]
bad = [{"timestamp": (base+timedelta(days=3, hours=i)).isoformat(), "chwLoadKw": v}
       for i, v in enumerate([-5.0, 1e9, 50.0])]

print("== ingestion ==")
for key, vendor, payload in (("DAIKIN_SITE", "vendor_a", a), ("CARRIER_SITE", "vendor_b", b)):
    r = c.post(f"/ingest/{key}", json={"vendor": vendor, "readings": payload})
    print(f"  {key:<14} {vendor:<9} -> {r.status_code} {r.json()}")
r = c.post("/ingest/DAIKIN_SITE", json={"vendor": "vendor_a", "readings": bad})
print(f"  含違規批次 -> {r.status_code} accepted={r.json()['accepted']} rejected={r.json()['rejected']}")
for e in r.json()["errors"]:
    print(f"     擋下 idx={e['index']} {e['field']}: {e['reason'][:60]}")

print("\n== 兩套方言歸一驗證 ==")
with service.SessionLocal() as s:
    for k in ("DAIKIN_SITE", "CARRIER_SITE"):
        site = s.scalar(sa.select(M.Site).where(M.Site.site_key == k))
        v = s.scalar(sa.select(M.QualityEvent.value_raw)
                     .where(M.QualityEvent.site_id == site.id,
                            M.QualityEvent.action == "flagged")
                     .order_by(M.QualityEvent.ts))
        print(f"  {k:<14} 第一筆 canonical 負荷 = {v:.2f} kW")

print("\n== 模型版本與排程 ==")
with service.SessionLocal() as s:
    for k in ("DAIKIN_SITE", "CARRIER_SITE"):
        mid = schedule.register_model_version(s, k, "lgbm_direct", datetime(2016,1,1),
                                              datetime(2016,12,31), 3.1, 0.68, "initial",
                                              trained_at=datetime(2016,12,31))
        res = schedule.daily_predict(s, k, base, [100.0+i for i in range(24)])
        print(f"  {k:<14} model#{mid} 每日預測寫入 {res.written} 列")
    res2 = schedule.daily_predict(s, "DAIKIN_SITE", base, [999.0]*24)
    n = s.scalar(sa.select(sa.func.count()).select_from(M.Prediction))
    print(f"  重放同一批 -> 總列數 {n}（冪等，應為 48 = 2 案場 × 24 步）")
    d = schedule.weekly_retrain_check(s, "DAIKIN_SITE", {"load_lag_1": 0.42},
                                      datetime(2017, 1, 15))
    print(f"  週檢查(冷卻期內): psi={d.max_psi} state={d.state} retrain={d.should_retrain} — {d.reason}")
    d = schedule.weekly_retrain_check(s, "DAIKIN_SITE", {"load_lag_1": 0.42},
                                      datetime(2017, 6, 1))
    print(f"  週檢查(冷卻期外): psi={d.max_psi} state={d.state} retrain={d.should_retrain} — {d.reason}")

html = c.get("/sites/DAIKIN_SITE/dashboard").text
Path(__file__).resolve().parents[1].joinpath("reports/p5_dashboard.html").write_text(html, encoding="utf-8")
print(f"\n儀表板落盤 reports/p5_dashboard.html（{len(html)} bytes）")
