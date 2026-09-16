import uuid
from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'caregiver', 'family', 'admin'
    status: Mapped[str] = mapped_column(String(20), default="ACTIVE", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    contextual_logins = relationship("ContextualLogin", back_populates="user", cascade="all, delete-orphan")
    care_logs = relationship("CareLog", back_populates="user", cascade="all, delete-orphan")
    session_logs = relationship("SessionActivityLog", back_populates="user", cascade="all, delete-orphan")
    trust_evaluations = relationship("TrustEvaluation", back_populates="user", cascade="all, delete-orphan")
    baseline = relationship("BehavioralBaseline", back_populates="user", uselist=False, cascade="all, delete-orphan")


class ContextualLogin(Base):
    __tablename__ = "contextual_logins"

    login_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False)
    device_fingerprint: Mapped[str] = mapped_column(String(255), nullable=False)
    geolocation: Mapped[str | None] = mapped_column(String(100), nullable=True)
    login_risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_successful: Mapped[bool] = mapped_column(Boolean, nullable=False)
    login_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    user = relationship("User", back_populates="contextual_logins")


class CareLog(Base):
    __tablename__ = "care_logs"

    log_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    patient_id: Mapped[str] = mapped_column(String(50), nullable=False)
    care_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_status: Mapped[str] = mapped_column(String(30), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    user = relationship("User", back_populates="care_logs")


class SessionActivityLog(Base):
    __tablename__ = "session_activity_logs"

    log_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    session_token: Mapped[str] = mapped_column(String(255), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_accessed: Mapped[str | None] = mapped_column(String(100), nullable=True)
    modification_volume: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    user = relationship("User", back_populates="session_logs")


class BehavioralBaseline(Base):
    __tablename__ = "behavioral_baselines"

    baseline_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), unique=True, nullable=False
    )
    avg_access_frequency: Mapped[float] = mapped_column(Float, nullable=False)
    out_of_hours_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    user = relationship("User", back_populates="baseline")


class TrustEvaluation(Base):
    __tablename__ = "trust_evaluations"

    evaluation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    perimeter_score: Mapped[float] = mapped_column(Float, nullable=False)
    behavioral_score: Mapped[float] = mapped_column(Float, nullable=False)
    unified_trust_score: Mapped[float] = mapped_column(Float, nullable=False)
    enforcement_action: Mapped[str] = mapped_column(String(30), nullable=False)  # 'ALLOW', 'STEP_UP_MFA', 'DENY'
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    user = relationship("User", back_populates="trust_evaluations")
    alerts = relationship("SecurityAlert", back_populates="evaluation", cascade="all, delete-orphan")


class SecurityAlert(Base):
    __tablename__ = "security_alerts"

    alert_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    evaluation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("trust_evaluations.evaluation_id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity_level: Mapped[str] = mapped_column(String(20), nullable=False)  # 'LOW', 'MEDIUM', 'HIGH'
    shap_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    evaluation = relationship("TrustEvaluation", back_populates="alerts")