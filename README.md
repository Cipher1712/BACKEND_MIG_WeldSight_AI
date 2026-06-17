# MIG-WeldSight AI - Backend (FastAPI on Railway)

All intelligence lives here: feature extraction, dynamic threshold, physics
classification, PCA/DBSCAN, training, traceability.

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

First frame the firmware should send is a setup frame:

```json
{"material": "mild_steel", "thickness_mm": 6}
```

Subsequent ingest frames:

```json
{"voltage": 24.7, "distance_mm": 125.4, "arc_on": true, "timestamp": 1750012345}
```