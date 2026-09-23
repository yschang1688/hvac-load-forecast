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
    signal: str = "psi"          # 觸發判定用的是哪個訊號："error"（P6 主訊號）或 "psi"（舊制）
    error_ratio: float = float("nan")


def weekly_retrain_check(s: Session, site_key: str, psi_by_col: dict[str, float],
                         now: datetime, psi_warn: float = 0.10, psi_alert: float = 0.25,
                         min_gap_days: int = 28, *,
                         skill_now: float | None = None,
                         skill_baseline: float | None = None,
                         error_ratio_alert: float = drift.ERROR_RATIO_ALERT) -> RetrainDecision:
    """每週漂移檢查。**觸發條件之外還有冷卻期**——沒有冷卻期，
    一段持續漂移會讓系統每週都重訓一次，而每次重訓都要重新驗收，
    運維成本與模型抖動都會失控。

    **P6 起主訊號是誤差，不是 PSI。** 傳入 `skill_now`（當週 skill）與 `skill_baseline`
    （基準期中位數，見 `drift.error_baseline`）時，以 `drift.error_verdict` 判定；
    PSI 仍會算並帶在回傳值裡，但只當「輸入端發生了什麼」的旁證，不觸發重訓。
    未傳誤差訊號時退回舊的 PSI 路徑——保留它是為了回放 P4 的實驗，
    不是因為它在這個場域有鑑別力（P4：624/624 週 alert）。
    """
    site = s.scalar(sa.select(M.Site).where(M.Site.site_key == site_key))
    if site is None:
        raise ValueError(f"未知案場 {site_key}")
    _, max_psi = drift.verdict(psi_by_col, psi_warn, psi_alert)
    last = s.scalar(sa.select(sa.func.max(M.ModelVersion.trained_at))
                    .where(M.ModelVersion.site_id == site.id))

    if skill_now is not None or skill_baseline is not None:
        state, ratio = drift.error_verdict(
            float("nan") if skill_now is None else skill_now,
            float("nan") if skill_baseline is None else skill_baseline,
            alert=error_ratio_alert)
        signal = "error"
        trigger_msg = f"誤差比 {ratio:.2f} 超過 {error_ratio_alert}（skill {skill_now:.3f} vs 基準 {skill_baseline:.3f}）" \
            if state == "alert" else f"誤差判定 {state}，未達觸發門檻"
    else:
        state, _ = drift.verdict(psi_by_col, psi_warn, psi_alert)
        ratio = float("nan")
        signal = "psi"
        trigger_msg = f"PSI={max_psi:.3f} 超過 {psi_alert}" if state == "alert" \
            else f"PSI 判定 {state}，未達觸發門檻"

    if state != "alert":
        return RetrainDecision(site_key, max_psi, state, False, trigger_msg, signal, ratio)
    if last is not None and (now - last) < timedelta(days=min_gap_days):
        left = min_gap_days - (now - last).days
        return RetrainDecision(site_key, max_psi, state, False,
                               f"在冷卻期內（還有 {left} 天）", signal, ratio)
    return RetrainDecision(site_key, max_psi, state, True, trigger_msg, signal, ratio)


def should_promote(old_val_skill: float | None, new_val_skill: float | None,
                   min_improvement: float = 0.0) -> tuple[bool, str]:
    """重訓後的晉升守門：新版本的驗收 skill 沒有比舊版本好就**不換**（回滾＝不晉升）。

    P4 的實驗裡重訓一律無條件上線；生產上這等於「每次重訓都是一次未驗收的部署」。
    這個守門是純函式，排程器在 `register_model_version` 之前呼叫它。
    舊版本沒有驗收指標（首版）時放行——否則系統永遠上不了第一個模型。
    """
    if old_val_skill is None or not np.isfinite(old_val_skill):
        return True, "無舊版驗收指標（首版），放行"
    if new_val_skill is None or not np.isfinite(new_val_skill):
        return False, "新版無驗收指標，不得晉升"
    if new_val_skill <= old_val_skill - min_improvement:
        return True, f"驗收 skill {old_val_skill:.3f} → {new_val_skill:.3f}，晉升"
    return False, f"驗收 skill {old_val_skill:.3f} → {new_val_skill:.3f} 未改善，保留舊版（回滾）"


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
