# Real-Time Movie Recommendation Platform — Claude Code Spec

## Project Overview

End-to-end movie recommendation system with real-time data ingestion, collaborative filtering, a served API, monitoring dashboard, and A/B testing infrastructure. Built to be fully Dockerized and deployed publicly (Railway or Render).

**Resume target:**
> Trained SVD collaborative filtering on MovieLens 1M (~1M ratings); deployed FastAPI (<50ms latency) with Docker, automated retraining pipeline, and hybrid cold-start solution (85%+ new user coverage). Built end-to-end data pipeline (schema validation, drift detection, offline/online evaluation: NDCG@10, HR@10) with CI/CD testing. Set up monitoring dashboard tracking availability, model accuracy, costs, and data drift. Designed A/B testing infrastructure comparing SVD vs. popularity-based cold-start.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Message Queue | Apache Kafka (via Docker) |
| Database | PostgreSQL 15 (via Docker) |
| ORM | SQLAlchemy + Alembic |
| ML | `scikit-surprise` (SVD), `pandas`, `numpy` |
| API | FastAPI + Uvicorn |
| Monitoring UI | Streamlit |
| Containerization | Docker + Docker Compose |
| Testing | pytest (target 80%+ coverage) |
| Deployment | Railway or Render (free tier) |
| Dataset | MovieLens 1M (`ml-1m`) |

---

## Repository Structure

```
movie-rec-platform/
├── CLAUDE.md                  # This file
├── README.md
├── docker-compose.yml         # Orchestrates all services
├── .env.example
│
├── data/                      # Raw + processed datasets
│   ├── download_movielens.py  # Downloads ml-1m from grouplens.org
│   └── ml-1m/                 # Gitignored, downloaded at setup
│
├── pipeline/                  # Kafka producer + consumer
│   ├── Dockerfile
│   ├── producer.py            # Simulates user interaction events
│   ├── consumer.py            # Consumes events, validates, stores to Postgres
│   ├── schema.py              # Pydantic event schema + validation
│   └── drift_detector.py      # Detects rating distribution drift over time
│
├── model/                     # Training + inference
│   ├── Dockerfile
│   ├── train.py               # SVD training on MovieLens data
│   ├── evaluate.py            # NDCG@10, HR@10 offline evaluation
│   ├── retrain_scheduler.py   # Triggers retraining on new data thresholds
│   └── model_store/           # Saved model artifacts (gitignored)
│
├── api/                       # FastAPI recommendation server
│   ├── Dockerfile
│   ├── main.py                # App entrypoint
│   ├── routers/
│   │   ├── recommend.py       # GET /recommend/{user_id}
│   │   ├── events.py          # POST /events (ingest interactions)
│   │   └── health.py          # GET /health
│   ├── services/
│   │   ├── svd_service.py     # Loads model, runs inference
│   │   ├── coldstart_service.py  # Popularity-based fallback
│   │   └── ab_service.py      # A/B flag routing logic
│   └── middleware/
│       └── latency.py         # Logs p50/p95 latency per request
│
├── monitoring/                # Streamlit dashboard
│   ├── Dockerfile
│   └── app.py                 # Live metrics: availability, drift, accuracy, A/B results
│
├── db/
│   ├── init.sql               # Initial schema creation
│   └── migrations/            # Alembic migration files
│
└── tests/
    ├── test_pipeline.py
    ├── test_api.py
    ├── test_model.py
    └── test_ab.py
```

---

## Database Schema

### `events` table — raw user interactions from Kafka
```sql
CREATE TABLE events (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    movie_id    INTEGER NOT NULL,
    event_type  VARCHAR(20) NOT NULL,  -- 'rating', 'click', 'watch'
    rating      FLOAT,                 -- nullable, only for 'rating' events
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ab_variant  VARCHAR(10),           -- 'svd' or 'coldstart'
    valid       BOOLEAN DEFAULT TRUE
);
```

### `recommendations` table — logged recommendations served
```sql
CREATE TABLE recommendations (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL,
    movie_ids       INTEGER[] NOT NULL,
    served_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    latency_ms      FLOAT,
    ab_variant      VARCHAR(10),
    model_version   VARCHAR(50)
);
```

### `model_runs` table — training history
```sql
CREATE TABLE model_runs (
    id          SERIAL PRIMARY KEY,
    version     VARCHAR(50) NOT NULL,
    trained_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ndcg_10     FLOAT,
    hr_10       FLOAT,
    n_ratings   INTEGER,
    rmse        FLOAT
);
```

### `drift_log` table — data drift tracking
```sql
CREATE TABLE drift_log (
    id              SERIAL PRIMARY KEY,
    logged_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metric          VARCHAR(50),   -- e.g. 'rating_mean', 'rating_stddev'
    baseline_value  FLOAT,
    current_value   FLOAT,
    drift_detected  BOOLEAN
);
```

---

## Service Specs

### 1. Kafka Producer (`pipeline/producer.py`)
- Simulates a stream of user interaction events by replaying MovieLens ratings in order
- Publishes JSON messages to topic `user-interactions`
- Event schema (validated via Pydantic in `schema.py`):
  ```json
  {
    "user_id": 123,
    "movie_id": 456,
    "event_type": "rating",
    "rating": 4.5,
    "timestamp": "2026-01-01T12:00:00Z"
  }
  ```
- Configurable replay speed via `PRODUCER_DELAY_MS` env var (default: 100ms)

### 2. Kafka Consumer (`pipeline/consumer.py`)
- Consumes from `user-interactions` topic
- Validates each event against Pydantic schema; marks invalid events with `valid=False`
- Writes all events (valid + invalid) to `events` table
- After every 10,000 new valid ratings, publishes a `retrain-trigger` message to topic `model-triggers`
- Runs drift detection every 1,000 events; logs results to `drift_log`

### 3. Model Training (`model/train.py`)
- Loads ratings from `events` table (or seeds from MovieLens 1M CSV on first run)
- Trains SVD model via `scikit-surprise` with cross-validation
- Saves model artifact to `model/model_store/svd_<version>.pkl`
- Logs NDCG@10, HR@10, RMSE to `model_runs` table
- Listens to `model-triggers` Kafka topic to auto-retrain

**SVD hyperparameters (starting point):**
```python
SVDConfig = {
    "n_factors": 100,
    "n_epochs": 20,
    "lr_all": 0.005,
    "reg_all": 0.02
}
```

### 4. FastAPI Server (`api/`)

#### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/recommend/{user_id}` | Returns top-N movie recommendations |
| POST | `/events` | Ingests a single interaction event |
| GET | `/health` | Returns service health + model version |
| GET | `/metrics` | Returns latency p50/p95, request count, error rate |

#### `GET /recommend/{user_id}` behavior
1. Check if `user_id` has ≥5 ratings in DB → if yes, use SVD variant; if no, use cold-start variant
2. A/B override: if `X-AB-Variant` header is set, use that variant instead
3. Log recommendation to `recommendations` table with latency, variant, model version
4. Return:
```json
{
  "user_id": 123,
  "variant": "svd",
  "model_version": "v1.2",
  "recommendations": [
    {"movie_id": 318, "title": "Shawshank Redemption", "predicted_rating": 4.7},
    ...
  ],
  "latency_ms": 23.4
}
```

#### Cold-Start Service (`api/services/coldstart_service.py`)
- Precomputes top-100 most popular movies (by rating count × avg rating) at startup
- Returns top-N from this list, excluding any movies the user has already rated
- Covers 85%+ of new users with zero model inference

#### A/B Service (`api/services/ab_service.py`)
- Assigns users to variants via `user_id % 2` (50/50 split)
- Variant `0` → SVD, Variant `1` → cold-start (for users who qualify for both)
- Logs variant assignment so `recommendations` table captures it
- Monitoring dashboard computes per-variant NDCG@10 from logged outcomes

### 5. Drift Detector (`pipeline/drift_detector.py`)
- Baseline = rating distribution from first 100K MovieLens events
- Monitors: `rating_mean`, `rating_stddev`, `ratings_per_minute`
- Flags drift if current window deviates >2 stddev from baseline
- Logs to `drift_log` table; dashboard visualizes trend

### 6. Monitoring Dashboard (`monitoring/app.py`)
Streamlit app with four tabs:

**Tab 1 — System Health**
- API availability (% uptime, last 100 health checks)
- Request volume over time (line chart)
- p50 / p95 latency (gauge)
- Error rate %

**Tab 2 — Model Performance**
- NDCG@10 and HR@10 over model versions (line chart)
- RMSE trend
- Last retrain timestamp + n_ratings used

**Tab 3 — Data Drift**
- `rating_mean` and `rating_stddev` over time vs. baseline
- Drift detected flags highlighted in red

**Tab 4 — A/B Testing**
- SVD vs. cold-start: NDCG@10 comparison
- Recommendation click-through rate by variant (simulated)
- Statistical significance note

---

## Docker Compose Services

```yaml
services:
  zookeeper:     # Required by Kafka
  kafka:         # Message broker, port 9092
  postgres:      # Database, port 5432
  producer:      # Replays MovieLens events into Kafka
  consumer:      # Reads Kafka, writes to Postgres, triggers retraining
  trainer:       # Trains SVD model, watches for retrain triggers
  api:           # FastAPI server, port 8000
  monitoring:    # Streamlit dashboard, port 8501
```

All services share a `rec-network` bridge network and use a shared `.env` file.

---

## Environment Variables (`.env.example`)

```env
POSTGRES_USER=rec_user
POSTGRES_PASSWORD=rec_pass
POSTGRES_DB=recommendations
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_INTERACTIONS_TOPIC=user-interactions
KAFKA_TRIGGER_TOPIC=model-triggers

PRODUCER_DELAY_MS=100
RETRAIN_THRESHOLD=10000

API_HOST=0.0.0.0
API_PORT=8000

MODEL_STORE_PATH=/app/model_store
```

---

## Evaluation Metrics

### Offline (computed during training)
- **NDCG@10** — normalized discounted cumulative gain at 10 recommendations (target: ~0.52)
- **HR@10** — hit rate at 10 (target: ~0.68)
- **RMSE** — root mean squared error on held-out test set

### Online (computed from `recommendations` + `events` tables)
- **Latency p95** — target <50ms
- **Availability** — target >99%
- **A/B delta NDCG@10** — difference between SVD and cold-start variants

---

## Development Order (Day by Day)

### Day 1 — Data Pipeline
1. `data/download_movielens.py` — download and parse `ml-1m`
2. `db/init.sql` — create all four tables
3. `pipeline/schema.py` — Pydantic event model
4. `pipeline/producer.py` — Kafka producer replaying MovieLens ratings
5. `pipeline/consumer.py` — Kafka consumer → Postgres
6. `docker-compose.yml` — wire up zookeeper, kafka, postgres, producer, consumer
7. Verify: events flowing into Postgres

### Day 2 — Model + API
1. `model/train.py` — SVD on MovieLens data, save artifact
2. `model/evaluate.py` — compute NDCG@10, HR@10
3. `api/services/svd_service.py` — load model, run inference
4. `api/services/coldstart_service.py` — popularity fallback
5. `api/routers/recommend.py` — GET /recommend/{user_id}
6. `api/routers/events.py` + `health.py`
7. `api/middleware/latency.py`
8. Test: curl /recommend/1 returns results in <50ms

### Day 3 — A/B Testing + Retraining + Drift
1. `api/services/ab_service.py` — variant routing
2. `pipeline/drift_detector.py`
3. `model/retrain_scheduler.py` — listens for Kafka trigger
4. `tests/` — pytest suite targeting 80%+ coverage
5. Integrate all services in docker-compose

### Day 4 — Monitoring + Deploy
1. `monitoring/app.py` — Streamlit dashboard (all four tabs)
2. `README.md` — architecture diagram, setup instructions, metrics
3. Deploy to Railway (or Render)
4. Record a short demo GIF for GitHub README
5. Write resume bullet

---

## Key Commands

```bash
# Start everything
docker compose up --build

# Download dataset
python data/download_movielens.py

# Run tests
pytest tests/ -v --cov=. --cov-report=term-missing

# Train model manually
docker compose exec trainer python model/train.py

# Hit the API
curl http://localhost:8000/recommend/1
curl http://localhost:8000/health
curl http://localhost:8000/metrics

# View monitoring dashboard
open http://localhost:8501
```

---

## Notes for Claude Code

- Always run `docker compose up postgres kafka` first before running any service locally outside Docker
- The `model_store/` directory must exist before the API starts — trainer must run first
- MovieLens 1M download requires ~25MB; gitignore `data/ml-1m/`
- Use `pytest-asyncio` for testing FastAPI async routes
- Keep all secrets in `.env`, never hardcode credentials
- Target <50ms p95 latency on `/recommend/{user_id}` — load model into memory at startup, don't reload per request
- SQLAlchemy sessions should be scoped per-request in FastAPI (use `Depends`)