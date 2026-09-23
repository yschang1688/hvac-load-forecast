"""P5｜SQLAlchemy ORM 綱要。

四張表對應「模型上線後要能回答的四個問題」：
- `sites`        ：這個案場是誰、用哪一份設定
- `predictions`  ：模型當時說了什麼（含發出時間與模型版本，可事後歸因）
- `quality_events`：閘門攔下了什麼（不是把髒資料悄悄丟掉）
- `model_versions`：這個版本是什麼時候、用哪段資料訓的，驗收指標多少

**設計決定**
1. `predictions` 同時記 `issued_at`（預測發出的時點）與 `target_ts`（被預測的時點）。
   只記其一就無法回答「這筆預測是提前幾小時做的」，也就無法歸因線上失準
   （P2 的教訓：特徵時戳與推論時戳分開記，才抓得到默默用舊資料的特徵）。
2. `(site_id, target_ts, horizon, model_version_id)` 唯一鍵＋UPSERT，
   使重放同一批預測不會產生重複列（冪等，同 dw-credit-star 的紀律）。
3. 補值與攔截**不刪原始值**，一律另存事件列。
"""
from __future__ import annotations
from datetime import datetime

from sqlalchemy import (DDL, Boolean, DateTime, Float, ForeignKey, Integer, String,
                        UniqueConstraint, Index, event)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Site(Base):
    __tablename__ = "sites"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    vendor: Mapped[str] = mapped_column(String(32))          # 廠牌方言，決定 ingestion 對映
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    # 銘牌額定容量。串流閘口只能檢查「有中繼資料可比對」的東西——
    # P1 的 R002 上限是從整段分布估的，逐筆進來時沒有分布可用，
    # 所以真實導入時上限必須來自設備銘牌而非資料本身（P1 報告已記此限制）。
    capacity_kw: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="site")
    events: Mapped[list["QualityEvent"]] = relationship(back_populates="site")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id"), index=True)
    algo: Mapped[str] = mapped_column(String(64))
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    train_start: Mapped[datetime] = mapped_column(DateTime)
    train_end: Mapped[datetime] = mapped_column(DateTime)
    # 驗收指標：沒有這兩欄就無法回答「這次重訓到底有沒有比較好」
    val_mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_skill: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrain_reason: Mapped[str] = mapped_column(String(128), default="initial")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (Index("ix_modelver_site_active", "site_id", "is_active"),)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime)      # 預測是何時發出的
    target_ts: Mapped[datetime] = mapped_column(DateTime)      # 被預測的時點
    horizon: Mapped[int] = mapped_column(Integer)              # target_ts - issued_at（小時）
    y_pred: Mapped[float] = mapped_column(Float)
    # 事後回填的實際值（＝actual_load）。由 `schedule.backfill_actuals` 從 `observations` 填入，
    # 不在寫預測時填——寫預測時實際值還不存在。
    y_true: Mapped[float | None] = mapped_column(Float, nullable=True)

    site: Mapped[Site] = relationship(back_populates="predictions")
    __table_args__ = (
        UniqueConstraint("site_id", "target_ts", "horizon", "model_version_id",
                         name="uq_pred_site_target_h_ver"),
        Index("ix_pred_site_issued", "site_id", "issued_at"),
    )


class QualityEvent(Base):
    __tablename__ = "quality_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime)
    rule_code: Mapped[str] = mapped_column(String(16))
    severity: Mapped[str] = mapped_column(String(8))
    value_raw: Mapped[float | None] = mapped_column(Float, nullable=True)
    action: Mapped[str] = mapped_column(String(32))            # blocked / imputed / flagged
    detail: Mapped[str] = mapped_column(String(256), default="")

    site: Mapped[Site] = relationship(back_populates="events")
    __table_args__ = (
        UniqueConstraint("site_id", "ts", "rule_code", name="uq_qe_site_ts_rule"),
        Index("ix_qe_site_ts", "site_id", "ts"),
    )


class Observation(Base):
    """實際量測值（canonical 單位 kW）。ingestion 閘口通過的讀數寫這裡。

    P5 版本只把通過的讀數記成 `quality_events` 的 ACCEPTED 事件列，實際值藏在 `value_raw`——
    能存但不能拿來算誤差：事件表的語意是「閘門做了什麼」，不是「負荷是多少」。
    P7 把實際值獨立成表，理由有二：(1) 回填 `predictions.y_true` 需要以 (site, ts) 精確對應；
    (2) seasonal-naive 基準線要取「上週同時刻」的實際值，同樣要從這裡取。
    """
    __tablename__ = "observations"
    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime)
    chw_load_kw: Mapped[float] = mapped_column(Float)
    __table_args__ = (UniqueConstraint("site_id", "ts", name="uq_obs_site_ts"),)


# ---------------------------------------------------------------------------
# 監控 view：預測誤差與週 skill（P7）
#
# skill ＝ 模型 MAE ÷ seasonal-naive MAE，與 P3／P4 同一個定義，**同一批列**上計算：
# 只有「實際值已回填」且「上週同時刻的實際值存在」的列才進分子與分母。
# 若改用 LEFT JOIN 再把缺的基準線當 0，分母會被灌水、skill 被壓低——看起來模型變好了，
# 其實是基準線被偷換成「預測 0」。守門測試 test_weekly_skill_excludes_rows_without_naive 釘這條。
#
# 刻意不用 MAPE：冰水負荷零膨脹，關機時分母趨近 0，MAPE 會爆掉（P3「分母退化」一節）。
# 用 CREATE OR REPLACE：metadata 的 after_create 在每次 create_all 都會觸發（服務每次啟動都呼叫）。
# 刻意不在 view 裡寫 CURRENT_DATE：歷史回放時牆鐘與模擬時點不同，
# 同 P5 `trained_at` 的牆鐘 bug。時間範圍一律由呼叫端以參數傳入。
# 週的切法是 PostgreSQL `date_trunc('week')`（週一起算），以 issued_at 歸週。
# 以「案場×週」彙總、跨模型版本合併：skill 是相對免費基準線的比值，不同版本可直接比；
# 監控要回答的是「這個案場最近是否比平常差」。**skill 的定義只寫在這裡一處**，
# Python 端只讀不重算——兩處定義時，一處寫錯另一處照樣綠燈（本專案的守門測試實際抓到過）。
# ---------------------------------------------------------------------------

_V_ERRORS = """
CREATE OR REPLACE VIEW v_prediction_errors AS
SELECT p.site_id, p.model_version_id, p.issued_at, p.target_ts, p.horizon,
       p.y_pred, p.y_true, o.chw_load_kw AS y_naive,
       abs(p.y_pred - p.y_true)      AS abs_err_model,
       abs(o.chw_load_kw - p.y_true) AS abs_err_naive
FROM predictions p
JOIN observations o
  ON o.site_id = p.site_id
 AND o.ts = p.target_ts - interval '168 hours'
WHERE p.y_true IS NOT NULL
"""

_V_WEEKLY = """
CREATE OR REPLACE VIEW v_weekly_skill AS
SELECT site_id,
       date_trunc('week', issued_at)                        AS week_start,
       count(DISTINCT model_version_id)                     AS n_versions,
       count(*)                                             AS n,
       avg(abs_err_model)                                   AS mae_model,
       avg(abs_err_naive)                                   AS mae_naive,
       avg(abs_err_model) / nullif(avg(abs_err_naive), 0)   AS skill
FROM v_prediction_errors
GROUP BY site_id, date_trunc('week', issued_at)
"""

event.listen(Base.metadata, "after_create", DDL(_V_ERRORS))
event.listen(Base.metadata, "after_create", DDL(_V_WEEKLY))
event.listen(Base.metadata, "before_drop", DDL("DROP VIEW IF EXISTS v_weekly_skill"))
event.listen(Base.metadata, "before_drop", DDL("DROP VIEW IF EXISTS v_prediction_errors"))
