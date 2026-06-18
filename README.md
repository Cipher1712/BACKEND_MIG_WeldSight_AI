# WeldSight AI Backend

Production voltage-only welding intelligence:

`750 Hz voltage -> 64/32 windowing -> 30 features -> physics -> EWMA +
Isolation Forest + physics-informed VAE -> defect classifier -> explanation ->
quality index -> WebSocket`

The 64-sample window covers 85.3 ms and produces a result every 42.7 ms.

## Deploy to Railway

1. Push the repo to GitHub (Lovable -> + menu -> GitHub -> Connect).
2. Railway -> **New Project -> Deploy from GitHub** -> select the repo, set
   the service root to `backend/`. Railway auto-detects the Dockerfile and
   `railway.toml`.
3. Add a **PostgreSQL** plugin; `DATABASE_URL` is injected automatically.
4. (Optional) Set `ALLOWED_ORIGINS=https://<your-frontend>.vercel.app`.
5. Deploy. Apply schema once if you prefer explicit migrations:
   `railway run psql $DATABASE_URL -f schema.sql`
   (Otherwise SQLAlchemy `create_all` runs at startup.)
5. Place exported artifacts in `models/`, or attach a Railway volume and set
   `MODEL_DIR`. Until artifacts are present, the service runs in an explicit
   physics fallback mode (`model_ready: false`).
6. Copy the public URL, e.g. `https://weldsight-api.up.railway.app`.

## Local development

```bash
cd backend
pip install -r requirements.txt
DATABASE_URL=sqlite:///./weldsight.db uvicorn app.main:app --reload --port 8000
```

## Endpoints

| Method | Path                                | Purpose                          |
|--------|-------------------------------------|----------------------------------|
| GET    | `/health`                           | Liveness probe                   |
| GET    | `/api/profiles`                     | List trained profiles            |
| GET    | `/api/profiles/{material}/{t}`      | Fetch one profile                |
| POST   | `/api/train`                        | Train baseline from good welds   |
| POST   | `/api/infer`                        | Batch inference (CSV upload)     |
| GET    | `/api/events?limit=200`             | Paginated anomaly history        |
| WS     | `/ws/stream`                        | ESP32 -> backend ingest          |
| WS     | `/ws/live`                          | Frontend <- backend live frames  |

## Offline model training

Datasets may be CSV/Parquet feature tables or JSON weld records. Feature tables
must contain the 30 names returned by `WindowFeatures.names()` and classifiers
also require a `label` column. Supported labels are:

`stable_arc`, `arc_instability`, `excessive_spatter`, `porosity_risk`,
`heat_input_high`, `heat_input_low`, `short_circuit_instability`,
`abnormal_arc_behaviour`, and `unknown_anomaly`.

```bash
python train_vae.py healthy_welds.json --output models
python train_classifier.py labeled_welds.json --output models
python evaluate.py held_out_welds.json --models models
python export_models.py --models models
pytest -q
```

Classifier training compares Random Forest, XGBoost, LightGBM, SVM, and MLP
with stratified cross-validation and randomized hyperparameter search. If an
optional training library is unavailable, the report records only the models
that were actually available.

First frame the firmware should send is a setup frame:

```json
{"material": "mild_steel", "thickness_mm": 6}
```

Subsequent ingest frames:

```json
{"voltage": 24.7, "distance_mm": 125.4, "arc_on": true, "timestamp": 1750012345}
```
