"""P7 守門測試：實際值回填、SQL 週 skill、誤差監控閉環。跑在真實 PostgreSQL。

每條測試對應一種「會靜默算錯、而且數字看起來合理」的情況。
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import sqlalchemy as sa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_service import _db_available  # noqa: E402

pytestmark = pytest.mark.skipif(not _db_available(), reason="PostgreSQL 未啟動")

EPOCH = datetime(2016, 1, 4)          # 週一


def _actual(ts: datetime) -> float:
    """實際負荷：偶數週 100、奇數週 110 → seasonal-naive（上週同時刻）的誤差恆為 10。"""
    return 100.0 + 10.0 * (((ts - EPOCH).days // 7) % 2)


@pytest.fixture()
def env():
    import db as M
    import service
    import schedule
    M.Base.metadata.drop_all(service.engine)
    M.Base.metadata.create_all(service.engine)
    with service.SessionLocal() as s:
        site = M.Site(site_key="MON", vendor="vendor_a", capacity_kw=1000)
        s.add(site); s.flush()
        mv = M.ModelVersion(site_id=site.id, algo="lgbm_direct", trained_at=datetime(2016, 1, 1),
                            train_start=datetime(2015, 1, 1), train_end=datetime(2015, 12, 31),
                            val_mae=1.0, val_skill=0.5, is_active=True)
        s.add(mv); s.commit()
        yield s, M, schedule, site.id, mv.id


def _obs(s, M, sid, start, end, fn=_actual):
    t, rows = start, []
    while t < end:
        rows.append({"site_id": sid, "ts": t, "chw_load_kw": fn(t)})
        t += timedelta(hours=1)
    s.execute(sa.insert(M.Observation), rows); s.commit()


def _preds(s, M, sid, mvid, week_start, model_err, fn=_actual):
    """該週每天 00:00 發一次 24 步預測，y_pred = 實際 + model_err。"""
    rows = []
    for d in range(7):
        iss = week_start + timedelta(days=d)
        for h in range(1, 25):
            tgt = iss + timedelta(hours=h)
            rows.append({"site_id": sid, "model_version_id": mvid, "issued_at": iss,
                         "target_ts": tgt, "horizon": h, "y_pred": fn(tgt) + model_err})
    s.execute(sa.insert(M.Prediction), rows); s.commit()


# ---- 回填 -------------------------------------------------------------------

def test_backfill_is_idempotent_and_respects_as_of(env):
    s, M, schedule, sid, mv = env
    w = datetime(2017, 3, 6)
    _obs(s, M, sid, w - timedelta(days=7), w + timedelta(days=8))
    _preds(s, M, sid, mv, w, 2.0)
    mid = w + timedelta(days=3)
    n1 = schedule.backfill_actuals(s, "MON", mid)
    assert 0 < n1 < 168, "as_of 之後的目標時點不得被回填（那時還不存在實際值）"
    later = s.scalar(sa.select(sa.func.count()).select_from(M.Prediction)
                     .where(M.Prediction.target_ts > mid, M.Prediction.y_true.is_not(None)))
    assert later == 0
    assert schedule.backfill_actuals(s, "MON", mid) == 0, "重跑同一個 as_of 必須更新 0 列（冪等）"
    assert schedule.backfill_actuals(s, "MON", w + timedelta(days=8)) == 168 - n1


def test_backfill_refills_when_actual_is_corrected(env):
    """實際值事後被更正（例如電表重送），已回填的列要跟著更新，不能停在舊值。"""
    s, M, schedule, sid, mv = env
    w = datetime(2017, 3, 6)
    _obs(s, M, sid, w - timedelta(days=7), w + timedelta(days=8))
    _preds(s, M, sid, mv, w, 2.0)
    schedule.backfill_actuals(s, "MON", w + timedelta(days=8))
    tgt = w + timedelta(hours=5)
    s.execute(sa.update(M.Observation).where(M.Observation.ts == tgt).values(chw_load_kw=999.0)); s.commit()
    assert schedule.backfill_actuals(s, "MON", w + timedelta(days=8)) >= 1
    assert s.scalar(sa.select(M.Prediction.y_true).where(M.Prediction.target_ts == tgt).limit(1)) == 999.0


# ---- SQL 週 skill -------------------------------------------------------------

def test_weekly_skill_matches_hand_computation(env):
    """模型誤差 2、基準線誤差 10 → skill 必須是 0.2。
    寫反（naive ÷ model）會得 5.0；用 MAPE 會得別的數——這條把定義釘死。"""
    s, M, schedule, sid, mv = env
    w = datetime(2017, 3, 6)
    _obs(s, M, sid, w - timedelta(days=7), w + timedelta(days=8))
    _preds(s, M, sid, mv, w, 2.0)
    schedule.backfill_actuals(s, "MON", w + timedelta(days=8))
    ws = schedule.weekly_skill(s, "MON", w + timedelta(days=8))
    assert len(ws) == 1 and ws[0]["n"] == 168
    assert ws[0]["skill"] == pytest.approx(0.2)


def test_weekly_skill_excludes_rows_without_naive(env):
    """上週同時刻沒有實際值的列，不得進分子也不得進分母。
    若用 LEFT JOIN 把缺的基準線當 0，分母會被灌水、skill 被壓低——模型看起來變好了。"""
    s, M, schedule, sid, mv = env
    w = datetime(2017, 3, 6)
    _obs(s, M, sid, w - timedelta(days=7), w + timedelta(days=8))
    _preds(s, M, sid, mv, w, 2.0)
    gone = [w - timedelta(days=7) + timedelta(hours=h) for h in range(1, 25)]   # 週一的「上週同時刻」
    s.execute(sa.delete(M.Observation).where(M.Observation.ts.in_(gone))); s.commit()
    s.execute(sa.update(M.Prediction).where(M.Prediction.target_ts <= w + timedelta(hours=24))
              .values(y_pred=10_000.0)); s.commit()                              # 這些列誤差巨大
    schedule.backfill_actuals(s, "MON", w + timedelta(days=8))
    r = schedule.weekly_skill(s, "MON", w + timedelta(days=8))[0]
    assert r["n"] == 168 - 24
    assert r["skill"] == pytest.approx(0.2), "沒有基準線的列混進來了"


def test_weekly_skill_only_returns_completed_weeks(env):
    s, M, schedule, sid, mv = env
    w = datetime(2017, 3, 6)
    _obs(s, M, sid, w - timedelta(days=7), w + timedelta(days=8))
    _preds(s, M, sid, mv, w, 2.0)
    schedule.backfill_actuals(s, "MON", w + timedelta(days=8))
    assert schedule.weekly_skill(s, "MON", w + timedelta(days=6)) == [], "未結束的週不得回報"


# ---- 閉環：週 skill → P6 誤差判定 ------------------------------------------------

def _history(s, M, sid, mv, weeks_errs):
    for w, e in weeks_errs:
        _preds(s, M, sid, mv, w, e)


def test_error_check_alerts_on_degradation_with_rolling_baseline(env):
    s, M, schedule, sid, mv = env
    cur = datetime(2017, 3, 27)
    _obs(s, M, sid, datetime(2017, 2, 27), cur + timedelta(days=8))
    _history(s, M, sid, mv, [(datetime(2017, 3, 6), 2.0), (datetime(2017, 3, 13), 2.0),
                             (datetime(2017, 3, 20), 2.0), (cur, 5.0)])
    r = schedule.weekly_error_check(s, "MON", cur + timedelta(days=7))
    assert r.baseline_mode == "rolling" and r.baseline == pytest.approx(0.2)
    assert r.skill_now == pytest.approx(0.5)
    assert r.decision.state == "alert" and r.decision.should_retrain is True


def test_error_check_stays_stable_when_error_is_normal(env):
    s, M, schedule, sid, mv = env
    cur = datetime(2017, 3, 27)
    _obs(s, M, sid, datetime(2017, 2, 27), cur + timedelta(days=8))
    _history(s, M, sid, mv, [(datetime(2017, 3, 6), 2.0), (datetime(2017, 3, 13), 2.0),
                             (datetime(2017, 3, 20), 2.0), (cur, 2.0)])
    r = schedule.weekly_error_check(s, "MON", cur + timedelta(days=7))
    assert r.decision.state == "stable" and r.decision.should_retrain is False


def test_same_period_baseline_beats_rolling_when_season_differs(env):
    """**反向會失敗的測試**：往年同期的 skill 本來就高（0.5，例如季節轉換期本來就難預測），
    最近幾週偏低（0.2）。今年同期 skill 0.4：
    - 同期基準 0.5 → 比值 0.8 → stable（正確：這個時期本來就這樣）
    - 若誤用滾動基準 0.2 → 比值 2.0 → alert → 無謂重訓
    這正是 P6 離線重算發現的「固定基準被季節性咬」。"""
    s, M, schedule, sid, mv = env
    cur = datetime(2017, 3, 27)
    _obs(s, M, sid, datetime(2016, 3, 7), datetime(2016, 4, 5))
    _obs(s, M, sid, datetime(2017, 2, 27), cur + timedelta(days=8))
    _history(s, M, sid, mv, [(datetime(2016, 3, 14), 5.0), (datetime(2016, 3, 21), 5.0),
                             (datetime(2016, 3, 28), 5.0),
                             (datetime(2017, 3, 6), 2.0), (datetime(2017, 3, 13), 2.0),
                             (datetime(2017, 3, 20), 2.0), (cur, 4.0)])
    r = schedule.weekly_error_check(s, "MON", cur + timedelta(days=7))
    assert r.baseline_mode == "same_period" and r.baseline_n == 3
    assert r.baseline == pytest.approx(0.5)
    assert r.decision.state == "stable" and r.decision.should_retrain is False


def test_shutdown_week_is_not_judged(env):
    """冰機整週關機：skill 分母退化，不得判定也不得觸發重訓。"""
    s, M, schedule, sid, mv = env
    cur = datetime(2017, 3, 27)
    off = lambda t: 0.0 if cur - timedelta(days=1) <= t < cur + timedelta(days=8) else _actual(t)
    _obs(s, M, sid, datetime(2017, 2, 27), cur + timedelta(days=8), fn=off)
    _history(s, M, sid, mv, [(datetime(2017, 3, 6), 2.0), (datetime(2017, 3, 13), 2.0),
                             (datetime(2017, 3, 20), 2.0)])
    _preds(s, M, sid, mv, cur, 3.0, fn=off)
    r = schedule.weekly_error_check(s, "MON", cur + timedelta(days=7))
    assert r.operating == "shutdown" and r.decision is None


def test_error_check_without_enough_history_does_not_pretend(env):
    s, M, schedule, sid, mv = env
    cur = datetime(2017, 3, 27)
    _obs(s, M, sid, datetime(2017, 3, 13), cur + timedelta(days=8))
    _history(s, M, sid, mv, [(datetime(2017, 3, 20), 2.0), (cur, 9.0)])
    r = schedule.weekly_error_check(s, "MON", cur + timedelta(days=7))
    assert r.decision is None and "不足" in r.reason


def test_ingest_writes_observations(env):
    """ingestion 閘口通過的讀數要落 observations，重送同一時點不產生重複列。"""
    from fastapi.testclient import TestClient
    import service
    s, M, schedule, sid, mv = env
    with TestClient(service.app) as c:
        body = {"vendor": "vendor_a", "readings": [
            {"timestamp": "2017-08-01T00:00:00", "chwLoadKw": 50.0, "oaTempC": 30.0, "dewPointC": 22.0},
            {"timestamp": "2017-08-01T01:00:00", "chwLoadKw": -1.0, "oaTempC": 30.0, "dewPointC": 22.0}]}
        c.post("/ingest/OBS_SITE", json=body)
        c.post("/ingest/OBS_SITE", json=body)
    site = s.scalar(sa.select(M.Site).where(M.Site.site_key == "OBS_SITE"))
    n = s.scalar(sa.select(sa.func.count()).select_from(M.Observation).where(M.Observation.site_id == site.id))
    assert n == 1, "負值被擋下不得落庫；重送不得重複"
