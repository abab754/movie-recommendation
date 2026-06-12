"""FastAPI application entrypoint."""

from collections import deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI

from api.middleware.latency import LatencyMiddleware
from api.routers import events, health, recommend
from api.services import coldstart_service, svd_service

# Shared latency storage so /metrics can access it
latency_store = deque(maxlen=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and precompute data at startup."""
    recommend.load_movie_metadata()

    try:
        svd_service.load_model()
        print("SVD model loaded successfully")
    except FileNotFoundError:
        print("WARNING: No SVD model found. Only cold-start will work until training runs.")

    try:
        coldstart_service.precompute_popular_movies()
    except Exception as e:
        print(f"WARNING: Could not precompute popular movies: {e}")

    yield


app = FastAPI(title="Movie Recommendation API", lifespan=lifespan)
app.add_middleware(LatencyMiddleware, latency_store=latency_store)
app.include_router(recommend.router)
app.include_router(events.router)
app.include_router(health.router)


@app.get("/metrics")
def metrics():
    """Return latency and request metrics."""
    if not latency_store:
        return {"p50_ms": 0, "p95_ms": 0, "request_count": 0}
    arr = np.array(latency_store)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
        "request_count": len(latency_store),
    }
