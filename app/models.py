from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Numeric, TIMESTAMP, JSON, UniqueConstraint, func, BigInteger
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Profile(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True)
    material = Column(String, nullable=False)
    thickness_mm = Column(Numeric(6, 2), nullable=False)
    learned_k = Column(Numeric(6, 3), nullable=False)
    mean_score = Column(Numeric(8, 4), nullable=False)
    std_score = Column(Numeric(8, 4), nullable=False)
    voltage_min = Column(Numeric(8, 3))
    voltage_max = Column(Numeric(8, 3))
    rms_min = Column(Numeric(8, 3))
    rms_max = Column(Numeric(8, 3))
    trained_windows = Column(Integer, nullable=False, default=0)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    __table_args__ = (UniqueConstraint("material", "thickness_mm", name="uq_profile_key"),)


class AnomalyEvent(Base):
    __tablename__ = "anomaly_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(TIMESTAMP(timezone=True), server_default=func.now())
    event_timestamp_ms = Column(BigInteger)
    material = Column(String, nullable=False)
    thickness_mm = Column(Numeric(6, 2), nullable=False)
    distance_mm = Column(Numeric(10, 3))
    distance_source = Column(String)
    anomaly_score = Column(Numeric(10, 4))
    threshold = Column(Numeric(10, 4))
    physics_label = Column(String)
    severity = Column(String)
    quality_index = Column(Integer)
    voltage_features = Column(JSON)
    recording_session_id = Column(String, ForeignKey("recording_sessions.session_id"), nullable=True)


class RecordingSession(Base):
    __tablename__ = "recording_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, unique=True, index=True)
    start_timestamp = Column(BigInteger, nullable=False)
    end_timestamp = Column(BigInteger)
    duration_ms = Column(Integer, nullable=False, default=0)
    sample_count = Column(Integer, nullable=False, default=0)
    sampling_rate_hz = Column(Numeric(10, 3))
    distance_mm = Column(Numeric(10, 3))
    distance_source = Column(String, nullable=False, default="Estimated")
    material = Column(String, nullable=False, default="mild_steel")
    thickness_mm = Column(Numeric(6, 2), nullable=False, default=6.0)
    trained = Column(Boolean, nullable=False, default=False)
    healthy_baseline = Column(Boolean, nullable=False, default=False)
    notes = Column(String)
    csv_path = Column(String)
    csv_size_bytes = Column(Integer, nullable=False, default=0)
    model_version_used = Column(String)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())


class TelemetrySample(Base):
    __tablename__ = "telemetry_samples"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("recording_sessions.session_id"), nullable=False, index=True)
    sample_index = Column(Integer, nullable=False)
    timestamp_ms = Column(BigInteger, nullable=False)
    voltage = Column(Numeric(12, 6), nullable=False)
    encoder_count = Column(Numeric(14, 4))
    distance_mm = Column(Numeric(10, 3), nullable=False)
    distance_source = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    __table_args__ = (UniqueConstraint("session_id", "sample_index", name="uq_sample_session_index"),)
