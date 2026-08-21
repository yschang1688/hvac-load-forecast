"""P5｜排程任務：每日預測與每週重訓檢查。

**為什麼是純函式而不是綁死 Airflow**：排程器是可換的（Airflow／APScheduler／cron／
雲端的 managed scheduler），但「每天做什麼、每週檢查什麼」不該跟著換。
兩個 job 寫成不依賴排程器的函式，排程器只負責觸發——換排程器時不用改業務邏輯，
也讓它們可以被單元測試直接呼叫（Airflow DAG 的邏輯如果寫在 operator 裡就測不到）。

冪等是排程的必要條件：排程器會重試，重試不能產生重複資料。
每日預測走 UPSERT、每週檢查只在真的觸發時才寫入新版本。
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

import db as M
import drift


@dataclass
class DailyResult:
    site_key: str
    issued_at: datetime
    written: int
    model_version_id: int | None


def daily_predict(s: Session, site_key: str, issued_at: datetime,
                  y_pred: list[float]) -> DailyResult:
    """每日預測：取當前啟用版本、寫入 24 步預測（UPSERT）。"""
    site = s.scalar(sa.select(M.Site).where(M.Site.site_key == site_key))
    if site is None:
        raise ValueError(f"未知案場 {site_key}")
    mv = s.scalar(sa.select(M.ModelVersion)
                  .where(M.ModelVersion.site_id == site.id, M.ModelVersion.is_active.is_(True))
                  .order_by(M.ModelVersion.trained_at.desc()))
    if mv is None:
        return DailyResult(site_key, issued_at, 0, None)
    rows = [{"site_id": site.id, "model_version_id": mv.id, "issued_at": issued_at,
             "target_ts": issued_at + timedelta(hours=h + 1), "horizon": h + 1,
             "y_pred": float(v)} for h, v in enumerate(y_pred)]
    stmt = pg_insert(M.Prediction).values(rows)
    s.execute(stmt.on_conflict_do_update(
        constraint="uq_pred_site_target_h_ver",
        set_={"y_pred": stmt.excluded.y_pred, "issued_at": stmt.excluded.issued_at}))
    s.commit()
    return DailyResult(site_key, issued_at, len(rows), mv.id)


@dataclass
class RetrainDecision:
    site_key: str
    max_psi: float
    state: str
    should_retrain: bool
    reason: str


def weekly_retrain_check(s: Session, site_key: str, psi_by_col: dict[str, float],
                         now: datetime, psi_warn: float = 0.10, psi_alert: float = 0.25,
                         min_gap_days: int = 28) -> RetrainDecision:
    """每週漂移檢查。**觸發條件之外還有冷卻期**——沒有冷卻期，
    一段持續漂移會讓系統每週都重訓一次，而每次重訓都要重新驗收，
    運維成本與模型抖動都會失控。"""
    site = s.scalar(sa.select(M.Site).where(M.Site.site_key == site_key))
    if site is None:
        raise ValueError(f"未知案場 {site_key}")
    state, max_psi = drift.verdict(psi_by_col, psi_warn, psi_alert)
    last = s.scalar(sa.select(sa.func.max(M.ModelVersion.trained_at))
                    .where(M.ModelVersion.site_id == site.id))
    if state != "alert":
        return RetrainDecision(site_key, max_psi, state, False, f"PSI 判定 {state}，未達觸發門檻")
    if last is not None and (now - last) < timedelta(days=min_gap_days):
        left = min_gap_days - (now - last).days
        return RetrainDecision(site_key, max_psi, state, False, f"在冷卻期內（還有 {left} 天）")
    return RetrainDecision(site_key, max_psi, state, True, f"PSI={max_psi:.3f} 超過 {psi_alert}")


def register_model_version(s: Session, site_key: str, algo: str, train_start: datetime,
                           train_end: datetime, val_mae: float | None, val_skill: float | None,
                           reason: str, trained_at: datetime | None = None) -> int:
    """註冊新版本並停用舊版本。**驗收指標必填**——沒有它就無法回答重訓有沒有變好。

    ⚠ `trained_at` 必須可注入。ORM 預設用 `utcnow()`（牆鐘時間），
    在部署回放或歷史模擬中會讓「距上次重訓多久」算成負數，
    冷卻期於是永遠成立、重訓永遠不觸發——而且不會報錯，只會安靜地什麼都不做。
    本專案的守門測試實際抓到過這個（模擬時點 2017，牆鐘 2026，算出「還有 3489 天」）。
    """
    site = s.scalar(sa.select(M.Site).where(M.Site.site_key == site_key))
    s.execute(sa.update(M.ModelVersion).where(M.ModelVersion.site_id == site.id)
              .values(is_active=False))
    mv = M.ModelVersion(site_id=site.id, algo=algo, train_start=train_start, train_end=train_end,
                        val_mae=val_mae, val_skill=val_skill, retrain_reason=reason, is_active=True,
                        trained_at=trained_at or train_end)
    s.add(mv); s.commit()
    return mv.id
