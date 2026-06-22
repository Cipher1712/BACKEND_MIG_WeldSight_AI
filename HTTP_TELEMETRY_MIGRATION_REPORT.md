# WeldSight AI HTTP Telemetry Migration Report

## Phase 1 Backend WebSocket Analysis

### WebSocket endpoints

| Endpoint | File | Purpose | Status |
| --- | --- | --- | --- |
| `/ws/stream` | `app/main.py` | ESP32 ingest path. Receives setup frames with `material` and `thickness_mm`, then voltage packets with `voltage`, `distance_mm`, `arc_on`, and `timestamp`. Buffers samples into 64-sample windows, extracts features, runs inference, persists anomaly events, and broadcasts live frames. | Deprecated, retained for rollback |
| `/ws/live` | `app/main.py` | Frontend live subscription path. Connected clients receive frames broadcast after telemetry inference. | Deprecated, retained for rollback |

### Files involved

| File | Role |
| --- | --- |
| `app/main.py` | FastAPI app, WebSocket routes, REST routes, training/profile/event routes, inference orchestration, anomaly persistence. |
| `app/features.py` | Window sizing, stride, feature extraction, and batch windowing. |
| `app/inference.py` | Main AI inference pipeline: physics assessment, anomaly scoring, quality index, labels, explanations, recommendations. |
| `app/physics.py` | Physics-based welding diagnosis and stability scoring. |
| `app/anomaly.py` | EWMA, Isolation Forest, and VAE-backed anomaly scoring. |
| `app/quality.py` | Quality index calculation. |
| `app/models.py` | SQLAlchemy profile and anomaly event models. |
| `app/db.py` | Database session and engine helpers used for anomaly persistence. |
| `tests/test_api.py` | Existing WebSocket contract test and new HTTP telemetry contract tests. |
| `requirements.txt` | Contains `websockets`, still required while WebSocket rollback paths remain. |

### Dependencies

The WebSocket ingest flow depends on FastAPI WebSocket support, `asyncio` broadcasting, JSON parsing, the shared `InferencePipeline`, feature extraction from `app.features`, database persistence through SQLAlchemy, and model artifacts under `models/`.

### Frontend dependencies

The current frontend can still use `/ws/live` to receive live frames. New polling-friendly endpoints are available for migration:

- `GET /telemetry/latest`
- `GET /metrics/latest`
- `GET /events/latest`

### AI pipeline dependencies

The preserved AI pipeline remains:

```text
telemetry
 -> 64-sample / 32-stride windowing
 -> feature extraction
 -> physics metrics
 -> anomaly detection
 -> classification/labels
 -> quality and explanation fields
```

## Files Modified

- `app/main.py`
- `app/telemetry_state.py`
- `tests/test_api.py`
- `HTTP_TELEMETRY_MIGRATION_REPORT.md`

## New Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/telemetry` | HTTP telemetry ingest for ESP32 packets. Validates payload, appends samples to the central telemetry buffer, runs available inference windows, persists anomalies, updates backend state, and broadcasts to existing live WebSocket subscribers during migration. |
| `GET` | `/telemetry/latest` | Returns latest accepted telemetry packet and latest inference frame when available. |
| `GET` | `/metrics/latest` | Returns latest quality, stability, anomaly, status, diagnosis, and model readiness metrics. |
| `GET` | `/events/latest` | Returns latest in-memory anomaly frames, falling back to persisted anomaly events when no in-memory events exist. |

`GET /health` already existed and now reports `status: healthy`.

## Deprecated Endpoints

- `WS /ws/stream`
- `WS /ws/live`

Both remain active and are marked with `# Deprecated` comments in `app/main.py` for rollback during migration verification.

## Telemetry Flow

```text
ESP32
 ↓ POST /telemetry
Backend
 ↓ shared telemetry buffer
feature extraction
 ↓
physics metrics
 ↓
anomaly detection
 ↓
classification/labels
 ↓
Backend State
 ↓
GET /telemetry/latest
GET /metrics/latest
GET /events/latest
```

During the transition, processed HTTP frames are also broadcast to `/ws/live` clients so the existing frontend can continue operating until it migrates to polling.

## Payload Contract

Expected HTTP ingest payload:

```json
{
  "voltage": [18.2, 18.3, 18.5],
  "distance_mm": 0,
  "arc_on": true,
  "timestamp": 123456
}
```

Optional compatibility fields are accepted:

- `material`, default `mild_steel`
- `thickness_mm`, default `6.0`

Malformed payloads return FastAPI/Pydantic structured `422` errors.

## Frontend Migration Readiness

The backend is ready for frontend migration from WebSocket live consumption to HTTP polling. The frontend can poll:

- `/telemetry/latest` for latest raw packet and inference frame
- `/metrics/latest` for dashboard metric widgets
- `/events/latest` for recent anomaly/event lists

The legacy WebSocket routes are still available until end-to-end HTTP polling is verified.
