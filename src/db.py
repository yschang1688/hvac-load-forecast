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

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer, String,
                        UniqueConstraint, Index)
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
    y_true: Mapped[float | None] = mapped_column(Float, nullable=True)  # 事後回填

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
