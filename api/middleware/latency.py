"""Middleware to track request latency (p50/p95)."""

import time
from collections import deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class LatencyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, latency_store: deque = None):
        super().__init__(app)
        self.latency_store = latency_store or deque(maxlen=1000)

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.latency_store.append(elapsed_ms)
        response.headers["X-Latency-Ms"] = f"{elapsed_ms:.1f}"

        return response
