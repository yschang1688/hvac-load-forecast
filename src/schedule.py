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
import pandas as pd
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


# ---------------------------------------------------------------------------
# P7｜實際值回填 → SQL 週 skill → P6 誤差監控
# ---------------------------------------------------------------------------

import operating_state as OS  # noqa: E402

_BACKFILL = sa.text("""
UPDATE predictions p
   SET y_true = o.chw_load_kw
  FROM observations o
 WHERE o.site_id = p.site_id
   AND o.ts = p.target_ts
   AND p.site_id = :site_id
   AND p.target_ts <= :as_of
   AND p.y_true IS DISTINCT FROM o.chw_load_kw
""")


def backfill_actuals(s: Session, site_key: str, as_of: datetime) -> int:
    """把 `observations` 的實際值回填到 `predictions.y_true`，回傳更新列數。

    冪等：`IS DISTINCT FROM` 讓第二次執行更新 0 列，實際值事後被更正時則會重填。
    `as_of` 由呼叫端傳入而非取牆鐘——歷史回放時兩者不同（P5 `trained_at` 的教訓）。
    """
    site = _site(s, site_key)
    n = s.execute(_BACKFILL, {"site_id": site.id, "as_of": as_of}).rowcount
    s.commit()
    return n


def weekly_skill(s: Session, site_key: str, until: datetime) -> list[dict]:
    """讀 `v_weekly_skill`：`until` 之前**已結束**的週（週一起算）。

    skill 的定義只在 view 裡（`db._V_WEEKLY`）；這裡只讀不重算。
    """
    site = _site(s, site_key)
    rows = s.execute(sa.text("""
        SELECT week_start, n, n_versions, mae_model, mae_naive, skill
          FROM v_weekly_skill
         WHERE site_id = :sid AND week_start + interval '7 days' <= :until
         ORDER BY week_start"""),
        {"sid": site.id, "until": until}).mappings().all()
    return [dict(r) for r in rows]


def _week_state(s: Session, site_id: int, week_start: datetime, cfg: dict) -> tuple[str, float]:
    """以觀測值判定該週的冰機運轉狀態；前後各帶 min_zero_run_hours 當 context，避免跨週停機被截斷。"""
    pad = timedelta(hours=int(cfg.get("min_zero_run_hours", OS.DEFAULTS["min_zero_run_hours"])))
    lo, hi = week_start - pad, week_start + timedelta(days=7) + pad
    obs = s.execute(sa.select(M.Observation.ts, M.Observation.chw_load_kw)
                    .where(M.Observation.site_id == site_id,
                           M.Observation.ts >= lo, M.Observation.ts < hi)).all()
    full = pd.date_range(lo, hi, freq="h", inclusive="left")
    ctx = pd.Series({r.ts: r.chw_load_kw for r in obs}, dtype=float).reindex(full)
    week = ctx[(ctx.index >= week_start) & (ctx.index < week_start + timedelta(days=7))]
    return OS.window_state(week, cfg, context=ctx)


@dataclass
class ErrorCheck:
    site_key: str
    week_start: datetime | None
    skill_now: float
    baseline: float
    baseline_mode: str           # same_period / rolling / none
    baseline_n: int
    operating: str               # operating / shutdown / unknown
    decision: RetrainDecision | None
    reason: str


def weekly_error_check(s: Session, site_key: str, as_of: datetime, cfg: dict | None = None) -> ErrorCheck:
    """每週誤差監控：回填 → 取上一個完整週的 skill → 判運轉狀態 → 算基準 → P6 判定。

    **基準（John 09-23 裁定：多年同期優先）**：
    - `same_period`：往年同一時期（±same_period_weeks 週）的週 skill 中位數。
      P6 離線重算發現固定早期基準會被季節性咬（冬季基準偏低，夏季誤差自然高於它）；
      同期基準比的是「今年這時候 vs 往年這時候」，季節被抵掉。
    - `rolling`：往年同期不足 min_baseline_weeks 週時，退回最近 rolling_weeks 週。
      抓突變快，但緩慢漂移會被吸收成新常態（P6 實測）。回傳值標明用的是哪一種。
    關機與 unknown 的週**不進基準、也不判定**：P3 已證明任何正規化子在關機視窗上都會退化。
    """
    c = {"min_rows": 72, "same_period_weeks": 3, "min_baseline_weeks": 3, "rolling_weeks": 8,
         "error_ratio_alert": drift.ERROR_RATIO_ALERT, "min_retrain_gap_days": 28,
         **OS.DEFAULTS, **(cfg or {})}
    site = _site(s, site_key)
    backfill_actuals(s, site_key, as_of)
    weeks = [w for w in weekly_skill(s, site_key, as_of)
             if w["n"] >= c["min_rows"] and w["skill"] is not None]
    if not weeks:
        return ErrorCheck(site_key, None, float("nan"), float("nan"), "none", 0, "unknown", None,
                          "沒有已回填且樣本足夠的完整週")
    states = {w["week_start"]: _week_state(s, site.id, w["week_start"], c)[0] for w in weeks}
    cur = weeks[-1]
    ws = cur["week_start"]
    if states[ws] != "operating":
        return ErrorCheck(site_key, ws, float(cur["skill"]), float("nan"), "none", 0, states[ws], None,
                          f"該週判為 {states[ws]}，不判定（關機週的 skill 分母退化）")

    hist = [w for w in weeks[:-1] if states[w["week_start"]] == "operating"]
    same = [w["skill"] for w in hist
            if (ws - w["week_start"]).days >= 330
            and _weeks_apart_mod_year(ws, w["week_start"]) <= c["same_period_weeks"]]
    if len(same) >= c["min_baseline_weeks"]:
        mode, base_vals = "same_period", same
    else:
        mode, base_vals = "rolling", [w["skill"] for w in hist[-c["rolling_weeks"]:]]
    base = drift.error_baseline(base_vals, min_weeks=c["min_baseline_weeks"])
    if not np.isfinite(base):
        return ErrorCheck(site_key, ws, float(cur["skill"]), base, mode, len(base_vals), "operating", None,
                          f"基準週數不足（{len(base_vals)} < {c['min_baseline_weeks']}），不判定")
    d = weekly_retrain_check(s, site_key, {}, as_of, min_gap_days=c["min_retrain_gap_days"],
                             skill_now=float(cur["skill"]), skill_baseline=base,
                             error_ratio_alert=c["error_ratio_alert"])
    return ErrorCheck(site_key, ws, float(cur["skill"]), base, mode, len(base_vals), "operating", d, d.reason)


def _weeks_apart_mod_year(a: datetime, b: datetime) -> float:
    """兩週在「年內位置」上相差幾週（跨年環繞），用來找往年同期。"""
    da = (a - b).days % 364
    return min(da, 364 - da) / 7


def _site(s: Session, site_key: str) -> M.Site:
    site = s.scalar(sa.select(M.Site).where(M.Site.site_key == site_key))
    if site is None:
        raise ValueError(f"未知案場 {site_key}")
    return site
