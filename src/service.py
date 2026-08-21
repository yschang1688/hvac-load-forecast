"""P5｜FastAPI 模型服務與 ingestion 閘口。

端點：
  POST /ingest/{site_key}   多廠牌讀數 → canonical → 品質閘 → 落庫（違規回 422 結構化錯誤）
  POST /predict/{site_key}  取當前啟用的模型版本產生 24 步預測並落庫（UPSERT 冪等）
  GET  /health              服務與資料庫連線
  GET  /sites/{site_key}/dashboard  預測 vs 實際、品質攔截、模型版本**同一頁**

**設計立場**：ingestion 失敗要回 422 並說明是哪個欄位、什麼值、違反哪條規則，
不是回 200 然後把 None 塞進資料庫——後者會讓髒資料變成模型的輸入，
而那時候已經看不出來它原本是什麼了。
"""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

import contracts as C
import db as M

DB_URL = os.environ.get(
    "HVAC_DB_URL", "postgresql+psycopg2://hvac:hvac_local_dev@localhost:55432/hvac")
engine = sa.create_engine(DB_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    M.Base.metadata.create_all(engine)
    yield


app = FastAPI(title="HVAC load forecast service", lifespan=lifespan)


class IngestBatch(BaseModel):
    vendor: str
    readings: list[dict]
    capacity_kw: float | None = None      # 首次見到該案場時登錄銘牌額定


class IngestResult(BaseModel):
    accepted: int
    rejected: int
    errors: list[dict]


@app.get("/health")
def health(s: Session = Depends(get_db)):
    s.execute(sa.text("select 1"))
    return {"status": "ok", "db": engine.url.get_backend_name()}


def _get_site(s: Session, site_key: str) -> M.Site:
    site = s.scalar(sa.select(M.Site).where(M.Site.site_key == site_key))
    if site is None:
        raise HTTPException(404, f"未知案場 {site_key}")
    return site


@app.post("/ingest/{site_key}", response_model=IngestResult)
def ingest(site_key: str, batch: IngestBatch, s: Session = Depends(get_db)) -> IngestResult:
    site = s.scalar(sa.select(M.Site).where(M.Site.site_key == site_key))
    if site is None:
        site = M.Site(site_key=site_key, vendor=batch.vendor, capacity_kw=batch.capacity_kw)
        s.add(site); s.flush()
    elif batch.capacity_kw is not None and site.capacity_kw is None:
        site.capacity_kw = batch.capacity_kw; s.flush()

    accepted, errors = 0, []
    for i, raw in enumerate(batch.readings):
        try:
            r = C.to_canonical(site_key, batch.vendor, raw, capacity_kw=site.capacity_kw)
        except C.ContractError as e:
            errors.append({"index": i, **e.as_dict()})
            # 攔截本身要留下紀錄：不是把髒資料悄悄丟掉
            s.execute(pg_insert(M.QualityEvent).values(
                site_id=site.id, ts=_safe_ts(raw, batch.vendor), rule_code="CONTRACT",
                severity="ERROR", value_raw=None, action="blocked",
                detail=e.as_dict()["reason"][:255]
            ).on_conflict_do_nothing(constraint="uq_qe_site_ts_rule"))
            continue
        accepted += 1
        s.execute(pg_insert(M.Prediction.__table__.metadata.tables["quality_events"]).values(
            site_id=site.id, ts=r.ts, rule_code="ACCEPTED", severity="INFO",
            value_raw=r.chw_load_kw, action="flagged", detail=""
        ).on_conflict_do_nothing(constraint="uq_qe_site_ts_rule"))
    s.commit()
    if accepted == 0 and errors:
        # 整批都不合法：回 422 並附逐筆理由，不要讓呼叫端以為成功了
        raise HTTPException(422, detail={"accepted": 0, "errors": errors[:20]})
    return IngestResult(accepted=accepted, rejected=len(errors), errors=errors[:20])


def _safe_ts(raw: dict, vendor: str) -> datetime:
    d = C.DIALECTS.get(vendor)
    v = raw.get(d.ts_field) if d else None
    try:
        return datetime.fromisoformat(str(v))
    except Exception:
        return datetime(1970, 1, 1)


class PredictRequest(BaseModel):
    issued_at: datetime
    horizon: int = 24
    y_pred: list[float]


@app.post("/predict/{site_key}")
def write_prediction(site_key: str, req: PredictRequest, s: Session = Depends(get_db)):
    """落庫採 UPSERT：同一批預測重放不會產生重複列（冪等）。"""
    site = _get_site(s, site_key)
    mv = s.scalar(sa.select(M.ModelVersion)
                  .where(M.ModelVersion.site_id == site.id, M.ModelVersion.is_active.is_(True))
                  .order_by(M.ModelVersion.trained_at.desc()))
    if mv is None:
        raise HTTPException(409, f"{site_key} 沒有啟用中的模型版本")
    if len(req.y_pred) != req.horizon:
        raise HTTPException(422, f"y_pred 長度 {len(req.y_pred)} 與 horizon {req.horizon} 不符")

    rows = [{"site_id": site.id, "model_version_id": mv.id, "issued_at": req.issued_at,
             "target_ts": req.issued_at + timedelta(hours=h + 1), "horizon": h + 1,
             "y_pred": float(v)} for h, v in enumerate(req.y_pred)]
    stmt = pg_insert(M.Prediction).values(rows)
    s.execute(stmt.on_conflict_do_update(
        constraint="uq_pred_site_target_h_ver",
        set_={"y_pred": stmt.excluded.y_pred, "issued_at": stmt.excluded.issued_at}))
    s.commit()
    return {"written": len(rows), "model_version_id": mv.id}


@app.get("/sites/{site_key}/dashboard", response_class=HTMLResponse)
def dashboard(site_key: str, s: Session = Depends(get_db)) -> str:
    """業務數字與品質狀態**同一頁**——不把稽核結果藏在附錄。"""
    site = _get_site(s, site_key)
    preds = s.execute(sa.select(M.Prediction).where(M.Prediction.site_id == site.id)
                      .order_by(M.Prediction.target_ts.desc()).limit(24)).scalars().all()
    blocked = s.scalar(sa.select(sa.func.count()).select_from(M.QualityEvent)
                       .where(M.QualityEvent.site_id == site.id,
                              M.QualityEvent.action == "blocked")) or 0
    accepted = s.scalar(sa.select(sa.func.count()).select_from(M.QualityEvent)
                        .where(M.QualityEvent.site_id == site.id,
                               M.QualityEvent.action == "flagged")) or 0
    mvs = s.execute(sa.select(M.ModelVersion).where(M.ModelVersion.site_id == site.id)
                    .order_by(M.ModelVersion.trained_at.desc()).limit(5)).scalars().all()
    rows = "".join(
        f"<tr><td>{p.target_ts:%Y-%m-%d %H:%M}</td><td>h+{p.horizon}</td>"
        f"<td>{p.y_pred:.1f}</td><td>{'' if p.y_true is None else f'{p.y_true:.1f}'}</td></tr>"
        for p in preds)
    vers = "".join(
        f"<tr><td>#{m.id}</td><td>{m.algo}</td><td>{m.trained_at:%Y-%m-%d}</td>"
        f"<td>{'' if m.val_skill is None else f'{m.val_skill:.3f}'}</td>"
        f"<td>{m.retrain_reason}</td></tr>" for m in mvs)
    total = accepted + blocked
    rate = (blocked / total * 100) if total else 0.0
    return f"""<!doctype html><meta charset="utf-8"><title>{site_key}</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:60rem}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}}
td,th{{border-bottom:1px solid #ddd;padding:.35rem .6rem;text-align:left;font-variant-numeric:tabular-nums}}
.k{{display:inline-block;margin-right:2rem}}.b{{color:#a33;font-weight:700}}</style>
<h1>{site_key}</h1>
<p><span class="k">收進 {accepted}</span><span class="k b">閘口擋下 {blocked}（{rate:.1f}%）</span>
<span class="k">模型版本 {len(mvs)}</span></p>
<h2>最近預測</h2><table><tr><th>目標時點</th><th>步長</th><th>預測</th><th>實際</th></tr>{rows}</table>
<h2>模型版本與驗收</h2><table><tr><th>版本</th><th>演算法</th><th>訓練於</th><th>val skill</th><th>重訓原因</th></tr>{vers}</table>
<p style="color:#666">品質狀態與業務數字同頁顯示；擋下的資料留有事件紀錄，不是被悄悄丟棄。</p>"""
