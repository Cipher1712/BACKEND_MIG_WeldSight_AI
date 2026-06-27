# WeldSight AI - MIG Welding Intelligence Backend

A production-ready FastAPI backend for real-time MIG welding quality monitoring. Ingests raw voltage telemetry from an ESP32 at 750 Hz, runs a multi-layer AI inference pipeline (physics engine + VAE + Isolation Forest), and exposes REST endpoints for a Vercel-hosted dashboard.

**Live deployment:** `https://backend-mig-weldsight-ai.onrender.com`  
**Frontend:** `https://mig-weld-sight-ai.vercel.app`

---

## System Architecture

```
ESP32 (750 Hz Voltage Sampling)
            │
            │ HTTPS POST /api/infer
            ▼
Render Backend
https://backend-mig-weldsight-ai.onrender.com
            │
            ▼
Feature Extraction
(30 physics-informed features, 64-sample / 32-stride windows)
            │
            ├── Physics Engine (arc stability, burn-through, cold arc, transfer)
            ├── VAE (reconstruction error - unsupervised anomaly)
            └── Isolation Forest (multivariate outlier scoring)
            │
            ▼
Adaptive Threshold Fusion + EWMA Smoothing
            │
            ▼
Risk Classification: Normal • Watch • Warning • Critical
            │
            ▼
SQLite / PostgreSQL Event Store
            │
            ▼
REST API (polling)
            │
            ▼
Vercel Dashboard
https://mig-weld-sight-ai.vercel.app
```

### Signal processing parameters

| Parameter | Value |
|---|---|
| Sampling rate | 750 Hz |
| Window size | 64 samples (85.3 ms) |
| Stride | 32 samples (42.7 ms cadence) |
| Features per window | 30 |

---

## Deployment

**Platform:** Render  
**Base URL:** `https://backend-mig-weldsight-ai.onrender.com`

The service is containerised via Docker and deployed directly from this repository. The `PORT` environment variable is set by Render automatically; the container entrypoint respects it:

```dockerfile
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

---

## API Reference

### Health Check

```
GET /health
```

Returns backend liveness and AI model readiness. Use this to confirm all artifacts are loaded before starting telemetry.

```json
{
  "status": "healthy",
  "model_ready": true,
  "vae_loaded": true,
  "scaler_loaded": true,
  "threshold_loaded": true,
  "isolation_forest_loaded": true,
  "window_size": 64,
  "stride": 32,
  "sampling_rate_hz": 750
}
```

### Inference (ESP32 → Backend)

```
POST /api/infer
```

The only endpoint the ESP32 writes to. Accepts a voltage packet, runs the full inference stack, persists any anomaly events, and updates backend state. Returns `405 Method Not Allowed` when accessed via GET (browser) - this is expected.

**Request body:**
```json
{
  "voltage": [18.2, 18.3, 18.5],
  "distance_mm": 125.4,
  "arc_on": true,
  "timestamp": 1750012345
}
```

Optional fields: `material` (default `mild_steel`), `thickness_mm` (default `6.0`). Malformed payloads return `422` with structured Pydantic validation errors.

### Polling Endpoints (Frontend → Backend)

| Method | Path | Description |
|---|---|---|
| `GET` | `/telemetry/latest` | Latest raw telemetry packet received from the ESP32 + latest inference frame. Returns `{}` before first packet. |
| `GET` | `/metrics/latest` | Latest quality index, stability score, anomaly score, risk label, diagnosis, and model readiness flags. Returns `{}` before first inference. |
| `GET` | `/events/latest` | Latest anomaly/risk events. Falls back to persisted DB events when no in-memory events exist. Returns `[]` initially. |

### Profile & Batch Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/profiles` | List all trained material/thickness weld profiles |
| `GET` | `/api/profiles/{material}/{thickness}` | Fetch a specific weld profile |
| `POST` | `/api/train` | Train a baseline model from a set of known-good welds |
| `POST` | `/api/infer` | Batch inference via CSV upload |
| `GET` | `/api/events?limit=200` | Paginated anomaly event history |

### WebSocket Endpoints (Deprecated - retained for rollback)

| Endpoint | Direction | Description |
|---|---|---|
| `WS /ws/stream` | ESP32 → Backend | Original ingest path |
| `WS /ws/live` | Backend → Frontend | Live frame broadcast |

The backend has migrated from WebSocket ingest to HTTP polling. During transition, processed HTTP frames are also broadcast to `/ws/live` subscribers. See [`HTTP_TELEMETRY_MIGRATION_REPORT.md`](./HTTP_TELEMETRY_MIGRATION_REPORT.md) for full migration details.

---

## Inference Pipeline Detail

```
POST /api/infer (voltage packet)
  ↓
Buffer samples into 64-sample windows (32-stride)
  ↓
Extract 30 features per window
  ↓
Physics Engine
  ├── Arc stability score
  ├── Burn-through risk (sustained high-voltage excursions)
  ├── Cold arc risk (voltage below nominal for material profile)
  └── Transfer irregularity (droplet transfer waveform pattern)
  ↓
EWMA smoothing
  ↓
Isolation Forest (multivariate outlier score)
  ↓
VAE reconstruction error (unsupervised anomaly score)
  ↓
Adaptive threshold fusion
  ↓
Quality index (0–100) + risk label + SHAP explanation
  ↓
Persist to event store if anomaly threshold exceeded
  ↓
Update in-memory state (polled by /metrics/latest, /events/latest)
```

When trained model artifacts are absent, the service automatically falls back to pure physics-based scoring with `model_ready: false` in the health response.

> **Note:** No supervised defect classifier is trained because the source data contains no trusted defect labels. All diagnoses are physics-derived.

---

## Project Structure

```
.
├── app/
│   ├── main.py            # FastAPI app, all routes, inference orchestration
│   ├── inference.py       # Full pipeline: physics → anomaly → quality → labels → SHAP
│   ├── features.py        # 64-sample windowing, stride, 30-feature extraction
│   ├── physics.py         # Physics-based welding diagnosis and stability scoring
│   ├── anomaly.py         # EWMA, Isolation Forest, and VAE anomaly scoring
│   ├── quality.py         # Quality index (0–100) calculation
│   ├── models.py          # SQLAlchemy ORM: weld profiles + anomaly events
│   ├── db.py              # Database session and engine helpers
│   └── telemetry_state.py # Shared in-memory telemetry buffer
├── data/
│   ├── MIG Sensor Data/   # Healthy voltage CSVs for VAE/IF training
│   └── Data_I*.csv        # Real-condition traces for evaluation
├── models/                # Trained artifacts (vae.pt, scaler.pkl, etc.)
├── scripts/
│   └── train_vae.py       # Offline training entry point
├── tests/
│   └── test_api.py        # HTTP + WebSocket contract tests
├── Dockerfile
├── schema.sql
├── requirements.txt
├── requirements-dev.txt
├── env.example
├── train_vae.py
├── evaluate.py
└── export_models.py
```

---

## Tech Stack

| Layer | Library / Version |
|---|---|
| API framework | FastAPI 0.115, Uvicorn 0.30.6 |
| Deep learning | PyTorch 2.5.1 (physics-informed VAE) |
| Anomaly detection | scikit-learn 1.5.1 (Isolation Forest) |
| Explainability | SHAP 0.46.0 |
| Gradient boosting | XGBoost 2.1.1, LightGBM 4.5.0 |
| Numerics | NumPy 1.26.4, SciPy 1.14.1, Pandas 2.2.3 |
| ORM | SQLAlchemy 2.0.35 + psycopg2-binary 2.9.9 |
| Validation | Pydantic 2.9.2 |
| WebSocket | websockets 13.0.1 |
| Container | Python 3.11-slim Docker image |

---

## Environment Variables

Copy `env.example` to `.env`:

```env
# PostgreSQL connection (Render injects this automatically when a Postgres add-on is attached)
# Use sqlite:///./weldsight.db for local development
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/weldsight

# CORS allowed origins - set to your Vercel frontend URL in production
ALLOWED_ORIGINS=https://mig-weld-sight-ai.vercel.app

# Path to trained model artifact directory
MODEL_DIR=models

# Logging verbosity: DEBUG | INFO | WARNING | ERROR
LOG_LEVEL=INFO
```

---

## Local Development

```bash
# Install runtime dependencies
pip install -r requirements.txt

# Install dev/test dependencies
pip install -r requirements-dev.txt

# Run locally with SQLite (no Postgres needed)
DATABASE_URL=sqlite:///./weldsight.db uvicorn app.main:app --reload --port 8000
```

Interactive API docs available at `http://localhost:8000/docs`.

---

## Docker

```bash
# Build
docker build -t weldsight-backend .

# Run against a local Postgres
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql+psycopg2://postgres:postgres@host.docker.internal:5432/weldsight \
  -e ALLOWED_ORIGINS=* \
  weldsight-backend
```

---

## Offline Model Training

Training uses only real voltage traces - no synthetic data.

**Supported voltage column aliases:** `MIGVoltage`, `MIG Voltage`, `Voltage`, `ArcVoltage`, `Voltage_V`, `MigVolatge` (observed typo in source files). Current, TIG, encoder, and label columns are ignored.

```bash
# Train VAE and Isolation Forest on healthy data
python scripts/train_vae.py --data data --output models --epochs 25

# Evaluate on real-condition traces
python scripts/evaluate_models.py --data data --models models

# Export serialised artifacts
python export_models.py --models models

# Run test suite
pytest -q
```

Generated artifacts:

| File | Description |
|---|---|
| `models/vae.pt` | Physics-informed Variational Autoencoder (PyTorch) |
| `models/scaler.pkl` | Feature scaler fitted on healthy training data |
| `models/isolation_forest.pkl` | Isolation Forest trained on healthy windows |
| `models/anomaly_threshold.json` | Adaptive threshold calibration values |

Place these under `MODEL_DIR` before deploying. Without them the service runs in physics-fallback mode (`model_ready: false`).

---

## Current Deployment Status

| Check | Status |
|---|---|
| Render deployment | ✅ Online |
| Health endpoint | ✅ Responding |
| VAE model | ✅ Loaded |
| Isolation Forest | ✅ Loaded |
| Adaptive thresholds | ✅ Loaded |
| Feature scaler | ✅ Loaded |
| Window size | ✅ 64 samples |
| Stride | ✅ 32 samples |
| Sampling rate | ✅ 750 Hz |
| Telemetry / metrics / events | ⏳ Populate once ESP32 starts transmitting |

---

## GitHub Repositories

| | URL |
|---|---|
| Backend | https://github.com/Cipher1712/BACKEND_MIG_WeldSight_AI |
| Frontend | https://github.com/Cipher1712/MIG-WeldSight_ai |

---

## License

Centre of Excellence in Advanced Manufacturing Technology, Indian Institute of Technology Kharagpur

