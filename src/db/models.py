from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from src.db.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="viewer")  # admin, analyst, viewer
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


class Flow(Base):
    __tablename__ = "flows"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    src_ip = Column(String(45), nullable=False, index=True)
    dst_ip = Column(String(45), nullable=False, index=True)
    src_port = Column(Integer, nullable=False)
    dst_port = Column(Integer, nullable=False)
    proto = Column(String(16), nullable=False)
    feature_vector = Column(JSONB, nullable=False)
    source = Column(String(16), nullable=False, default="offline")  # live | offline

    alert = relationship("Alert", back_populates="flow", uselist=False)

    __table_args__ = (
        Index("ix_flows_ts_src_dst", "ts", "src_ip", "dst_ip"),
    )


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    flow_id = Column(Integer, ForeignKey("flows.id", ondelete="CASCADE"), nullable=False, index=True)
    level = Column(String(24), nullable=False, index=True)  # benign | suspicious | confirmed_malicious
    attack_class = Column(String(64), nullable=True, index=True)
    confidence = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=True)
    model_version = Column(String(64), nullable=False)
    status = Column(String(24), nullable=False, default="open")  # open | true_positive | false_positive
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    flow = relationship("Flow", back_populates="alert")
    evidence = relationship("AlertEvidence", back_populates="alert", uselist=False, cascade="all, delete-orphan")


class AlertEvidence(Base):
    __tablename__ = "alert_evidence"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id", ondelete="CASCADE"), unique=True, nullable=False)
    shap_values = Column(JSONB, nullable=False)
    detector_scores = Column(JSONB, nullable=False)
    notes = Column(Text, nullable=True)

    alert = relationship("Alert", back_populates="evidence")


class ModelRegistry(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(64), unique=True, nullable=False, index=True)
    type = Column(String(32), nullable=False)  # xgboost | random_forest | isolation_forest
    metrics = Column(JSONB, nullable=False)
    feature_schema = Column(String(64), nullable=False, default="v1.0.0")
    trained_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    active = Column(Boolean, default=False, nullable=False, index=True)


class CaptureSession(Base):
    __tablename__ = "capture_sessions"

    id = Column(Integer, primary_key=True, index=True)
    interface = Column(String(32), nullable=False)
    started_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    started_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    stopped_at = Column(DateTime(timezone=True), nullable=True)
    authorised_scope = Column(Text, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(64), nullable=False, index=True)
    target = Column(String(128), nullable=True)
    details = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
