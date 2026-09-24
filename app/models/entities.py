import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, JSON
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
    mfa_code = Column(String(6), nullable=True)
    mfa_expires_at = Column(DateTime, nullable=True)

    # Relationships
    contextual_logins = relationship("ContextualLogin", back_populates="user", cascade="all, delete-orphan")
    care_logs = relationship("CareLog", back_populates="user", cascade="all, delete-orphan")
    session_logs = relationship("SessionActivityLog", back_populates="user", cascade="all, delete-orphan")
    trust_evaluations = relationship("TrustEvaluation", back_populates="user", cascade="all, delete-orphan")
    baseline = relationship("BehavioralBaseline", back_populates="user", uselist=False, cascade="all, delete-orphan")
    behavioral_audit_logs = relationship("BehavioralAuditLog", back_populates="user", cascade="all, delete-orphan")


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


class PendingRegistration(Base):
    __tablename__ = "pending_registrations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="caregiver", nullable=False)
    otp_code = Column(String(6), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class UserDeviceProfile(Base):
    __tablename__ = "user_device_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    device_fingerprint = Column(String(128), nullable=False, index=True)

    # lifecycle state: 'NEW', 'PROBATIONARY', 'TRUSTED', 'REVOKED
    trust_state = Column(String(32), default="NEW", nullable=False)
    successful_login_count = Column(Integer, default=0, nullable=False)

    # Hardware metadata for security audits & UI display
    device_model = Column(String(64), nullable=True)
    os_name = Column(String(32), nullable=True)

    # Verification Timestamps
    first_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_mfa_completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("user_id", "device_fingerprint", name="uq_user_device_fingerprint"),
    )


class BehavioralAuditLog(Base):
    __tablename__ = "behavioral_audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False)
    session_revocation_recommended: Mapped[bool] = mapped_column(Boolean, nullable=False)
    risk_factors: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    telemetry_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    user = relationship("User", back_populates="behavioral_audit_logs")


class ElderlyProfile(Base):
    __tablename__ = "elderly_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # User.user_id is UUID in the existing schema; keep that table unchanged.
    family_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True
    )
    full_name: Mapped[str] = mapped_column(String(100), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str | None] = mapped_column(String(20), nullable=True)
    care_tier: Mapped[str] = mapped_column(String(50), default="Daily Assistance", nullable=False)
    chronic_conditions: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    emergency_contact_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    emergency_contact_phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    residence_address: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )


class CareSession(Base):
    __tablename__ = "care_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    elderly_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("elderly_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    family_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True
    )
    caregiver_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True, index=True
    )
    scheduled_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    service_tier: Mapped[str] = mapped_column(String(50), nullable=False)
    special_instructions: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="REQUESTED", nullable=False, index=True)
    vitals_recorded: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    encounter_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
