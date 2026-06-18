import os
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./weldsight.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def repair_sqlite_autoincrement_tables() -> None:
    """Repair legacy SQLite tables whose PK type prevents id autogeneration."""
    if engine.dialect.name != "sqlite":
        return
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(anomaly_events)")).mappings().all()
        if not rows:
            return
        id_col = next((row for row in rows if row["name"] == "id"), None)
        if id_col is None or str(id_col["type"]).upper() == "INTEGER":
            return
        conn.execute(text("DROP INDEX IF EXISTS idx_events_ts"))
        conn.execute(text("DROP INDEX IF EXISTS idx_events_profile"))
        conn.execute(text("""
            CREATE TABLE anomaly_events_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                material VARCHAR NOT NULL,
                thickness_mm NUMERIC(6, 2) NOT NULL,
                distance_mm NUMERIC(10, 3),
                anomaly_score NUMERIC(10, 4),
                threshold NUMERIC(10, 4),
                physics_label VARCHAR,
                severity VARCHAR,
                quality_index INTEGER,
                voltage_features JSON
            )
        """))
        conn.execute(text("""
            INSERT INTO anomaly_events_new (
                id, ts, material, thickness_mm, distance_mm, anomaly_score,
                threshold, physics_label, severity, quality_index, voltage_features
            )
            SELECT
                id, ts, material, thickness_mm, distance_mm, anomaly_score,
                threshold, physics_label, severity, quality_index, voltage_features
            FROM anomaly_events
            WHERE id IS NOT NULL
        """))
        conn.execute(text("DROP TABLE anomaly_events"))
        conn.execute(text("ALTER TABLE anomaly_events_new RENAME TO anomaly_events"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_events_ts ON anomaly_events (ts DESC)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_events_profile ON anomaly_events (material, thickness_mm)"))


@contextmanager
def get_session() -> Session:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
