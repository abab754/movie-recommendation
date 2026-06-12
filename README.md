# Real-Time Movie Recommendation Platform

End-to-end movie recommendation system with real-time data ingestion, SVD collaborative filtering, a low-latency serving API, automated retraining, drift detection, a monitoring dashboard, and A/B testing infrastructure. Fully Dockerized.

## Live Demo

- **API**: https://api-production-a0364.up.railway.app — try [`/recommend/2`](https://api-production-a0364.up.railway.app/recommend/2), [`/health`](https://api-production-a0364.up.railway.app/health), [`/docs`](https://api-production-a0364.up.railway.app/docs)
- **Monitoring Dashboard**: https://monitoring-production-4b75.up.railway.app

Deployed on Railway as a slim stack (Postgres seeded with 200K MovieLens events + API with the trained model baked in + dashboard). The full Kafka pipeline below runs locally via Docker Compose.

## Architecture

```
                         ┌──────────────────────────────────────────────┐
                         │                    KAFKA                     │
 ratings.dat ──► Producer ──► [user-interactions] ──► Consumer ──► PostgreSQL
                                      ▲                    │           ▲ │
 POST /events ────────────────────────┘                    │           │ │
                                              every 10K ratings        │ │
                                                           ▼           │ │
                                                  [model-triggers]     │ │
                                                           │           │ │
                                                           ▼           │ │
                                               Trainer (SVD retrain) ──┘ │
                                                           │             │
                                                           ▼             │
                                                  model-store volume     │
                                                           │             │
                                                           ▼             │
 GET /recommend/{id} ──────────────────────────► FastAPI ──┴─────────────┘
                                                    ▲
 Streamlit Dashboard ───────────────────────────────┘
```

- **Producer** replays MovieLens 1M ratings as a live event stream
- **Consumer** validates events (Pydantic), stores them, runs drift detection, and triggers retraining every 10K ratings
- **Trainer** retrains SVD on accumulated data and publishes versioned model artifacts
- **API** serves recommendations: SVD for users with history, popularity-based cold-start for new users, with A/B split between variants
- **Dashboard** tracks availability, latency, model metrics, drift, and A/B results

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Message Queue | Apache Kafka |
| Database | PostgreSQL 15 |
| ML | scikit-surprise (SVD) |
| API | FastAPI + Uvicorn |
| Monitoring | Streamlit |
| Containerization | Docker Compose |
| CI | GitHub Actions (pytest, 80%+ coverage gate) |
| Dataset | MovieLens 1M (~1M ratings, 6K users, 3.9K movies) |

## Quick Start

```bash
# 1. Download the dataset (~25MB)
python data/download_movielens.py

# 2. Configure environment
cp .env.example .env

# 3. Start all services
docker compose up --build

# 4. Train the initial model
docker compose exec trainer python -m model.train
docker compose restart api
```

Then:
- API: http://localhost:8000 (`/recommend/1`, `/health`, `/metrics`, docs at `/docs`)
- Dashboard: http://localhost:8501

## API

| Method | Path | Description |
|---|---|---|
| GET | `/recommend/{user_id}` | Top-N recommendations (SVD or cold-start) |
| POST | `/events` | Ingest an interaction event (goes through Kafka) |
| GET | `/health` | Service health + model version |
| GET | `/metrics` | p50/p95 latency, request count |

Example:

```bash
curl http://localhost:8000/recommend/2
```

```json
{
  "user_id": 2,
  "variant": "svd",
  "model_version": "v20260612_192709",
  "recommendations": [
    {"movie_id": 919, "predicted_rating": 4.53, "title": "Wizard of Oz, The (1939)"},
    ...
  ],
  "latency_ms": 40.4
}
```

Force a variant with the `X-AB-Variant: svd|coldstart` header.

## Evaluation

**Offline** (logged to `model_runs` on every training run):
- NDCG@10, HR@10 on a held-out test set
- RMSE

**Online** (computed from `recommendations` + `events` tables):
- p95 serving latency (target <50ms, model held in memory)
- A/B comparison: SVD vs popularity-based cold-start

## Testing

```bash
pytest tests/ -v --cov=pipeline --cov=model --cov=api
```

55 tests, 97% coverage (integration loops requiring live Kafka/Postgres are
excluded via `.coveragerc` and covered by the running stack instead).
CI runs the suite on every push with an 80% coverage gate.

## Project Structure

```
├── data/              # MovieLens download script (dataset gitignored)
├── db/                # Postgres schema (events, recommendations, model_runs, drift_log)
├── pipeline/          # Kafka producer, consumer, event schema, drift detector
├── model/             # SVD training, evaluation, retrain scheduler
├── api/               # FastAPI app: routers, services (svd/coldstart/ab), middleware
├── monitoring/        # Streamlit dashboard (5 tabs incl. live demo)
└── tests/             # pytest suite
```

## Design Notes

- **Cold-start**: top-100 popular movies (rating count × avg rating) precomputed at API startup; serves new users with zero model inference
- **A/B testing**: deterministic 50/50 split via `user_id % 2`; every served recommendation is logged with its variant for offline comparison
- **Drift detection**: baseline from first 100K ratings; flags windows deviating >2σ on rating mean/stddev
- **Retraining**: consumer publishes a Kafka trigger every 10K valid ratings; trainer retrains and atomically swaps `svd_latest.pkl`
