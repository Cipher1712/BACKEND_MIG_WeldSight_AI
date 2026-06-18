from sqlalchemy import Column, Integer, String, Numeric, TIMESTAMP, JSON, UniqueConstraint, func
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
    material = Column(String, nullable=False)
    thickness_mm = Column(Numeric(6, 2), nullable=False)
    distance_mm = Column(Numeric(10, 3))
    anomaly_score = Column(Numeric(10, 4))
    threshold = Column(Numeric(10, 4))
    physics_label = Column(String)
    severity = Column(String)
    quality_index = Column(Integer)
    voltage_features = Column(JSON)
