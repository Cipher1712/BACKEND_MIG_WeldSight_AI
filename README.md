# WeldSight AI Backend

Production voltage-only welding intelligence:

`750 Hz voltage -> validation -> 64/32 windowing -> 30 features -> physics ->
EWMA -> Isolation Forest -> physics-informed VAE -> quality index ->
explainability -> WebSocket`

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
5. Train or place exported artifacts in `models/`, or attach a Railway volume
   and set `MODEL_DIR`. Until artifacts are present, the service runs in an
   explicit physics fallback mode (`model_ready: false`).
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

## Offline model training from real data only

The training script uses only real voltage traces from:

- `data/MIG Sensor Data/*.csv` for healthy VAE/Isolation Forest training
- `data/Data_I*.csv` for real-condition evaluation

Only voltage columns are read. Supported voltage aliases are `MIGVoltage`,
`MIG Voltage`, `Voltage`, `ArcVoltage`, `Voltage_V`, and the observed
`MigVolatge` typo in the provided files. Current, TIG, encoder, and any label
columns are ignored.

```bash
python scripts/train_vae.py --data data --output models --epochs 25
python scripts/evaluate_models.py --data data --models models
python export_models.py --models models
pytest -q
```

Generated artifacts:

- `models/vae.pt`
- `models/scaler.pkl`
- `models/isolation_forest.pkl`
- `models/anomaly_threshold.json`

No supervised defect classifier is trained because the provided data has no
trusted defect labels. Diagnoses are physics-based, for example arc instability,
burn-through risk, cold arc risk, and transfer irregularity.

First frame the firmware should send is a setup frame:

```json
{"material": "mild_steel", "thickness_mm": 6}
```

Subsequent ingest frames:

```json
{"voltage": 24.7, "distance_mm": 125.4, "arc_on": true, "timestamp": 1750012345}
```
