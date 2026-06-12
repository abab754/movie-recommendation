"""Tests for the FastAPI endpoints with mocked DB and services."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routers import recommend
from api.services import coldstart_service, svd_service


@pytest.fixture
def client(monkeypatch):
    # Skip startup work that needs real DB/files/model
    monkeypatch.setattr(recommend, "load_movie_metadata", lambda: None)
    monkeypatch.setattr(svd_service, "load_model", lambda: None)
    monkeypatch.setattr(
        coldstart_service, "precompute_popular_movies", lambda: None
    )

    # Fake movie metadata
    monkeypatch.setattr(recommend, "_all_movie_ids", [1, 2, 3, 4, 5])
    monkeypatch.setattr(
        recommend,
        "_movie_titles",
        {i: f"Movie {i}" for i in range(1, 6)},
    )

    # No-op recommendation logging
    monkeypatch.setattr(
        recommend, "log_recommendation", lambda *args, **kwargs: None
    )

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_user_with_history(monkeypatch):
    """User has rated 5+ movies -> qualifies for SVD."""
    monkeypatch.setattr(
        recommend, "get_user_rated_movies", lambda user_id: {1, 2, 3, 4, 5}
    )


@pytest.fixture
def mock_new_user(monkeypatch):
    """User has no ratings -> cold-start."""
    monkeypatch.setattr(recommend, "get_user_rated_movies", lambda user_id: set())


@pytest.fixture
def mock_services(monkeypatch):
    monkeypatch.setattr(
        svd_service,
        "get_recommendations",
        lambda user_id, all_ids, rated, n=10: [
            {"movie_id": 1, "predicted_rating": 4.5}
        ],
    )
    monkeypatch.setattr(svd_service, "get_model_version", lambda: "v_test")
    monkeypatch.setattr(
        coldstart_service,
        "get_recommendations",
        lambda rated, n=10: [{"movie_id": 2, "predicted_rating": 4.0}],
    )


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert "model_version" in body


class TestMetricsEndpoint:
    def test_metrics_returns_stats(self, client):
        # Make a request first so there's at least one latency sample
        client.get("/health")
        response = client.get("/metrics")
        assert response.status_code == 200
        body = response.json()
        assert "p50_ms" in body
        assert "p95_ms" in body
        assert body["request_count"] >= 1

    def test_latency_header_present(self, client):
        response = client.get("/health")
        assert "x-latency-ms" in response.headers


class TestRecommendEndpoint:
    def test_user_with_history_even_id_gets_svd(
        self, client, mock_user_with_history, mock_services
    ):
        response = client.get("/recommend/2")
        assert response.status_code == 200
        body = response.json()
        assert body["variant"] == "svd"
        assert body["model_version"] == "v_test"
        assert len(body["recommendations"]) == 1
        assert body["recommendations"][0]["title"] == "Movie 1"

    def test_user_with_history_odd_id_gets_coldstart(
        self, client, mock_user_with_history, mock_services
    ):
        response = client.get("/recommend/3")
        assert response.status_code == 200
        assert response.json()["variant"] == "coldstart"

    def test_new_user_gets_coldstart(self, client, mock_new_user, mock_services):
        response = client.get("/recommend/2")  # even, but no history
        assert response.status_code == 200
        assert response.json()["variant"] == "coldstart"

    def test_ab_header_override(
        self, client, mock_user_with_history, mock_services
    ):
        # User 3 is odd (would get coldstart), header forces svd
        response = client.get("/recommend/3", headers={"X-AB-Variant": "svd"})
        assert response.status_code == 200
        assert response.json()["variant"] == "svd"

    def test_response_includes_latency(
        self, client, mock_new_user, mock_services
    ):
        response = client.get("/recommend/1")
        body = response.json()
        assert "latency_ms" in body
        assert body["latency_ms"] >= 0

    def test_invalid_user_id_rejected(self, client):
        response = client.get("/recommend/not_a_number")
        assert response.status_code == 422


class TestEventsEndpoint:
    def test_valid_event_accepted(self, client, monkeypatch):
        from api.routers import events as events_router

        class FakeProducer:
            def produce(self, *args, **kwargs):
                pass

            def flush(self):
                pass

        monkeypatch.setattr(events_router, "_get_producer", lambda: FakeProducer())

        response = client.post(
            "/events",
            json={
                "user_id": 1,
                "movie_id": 318,
                "event_type": "rating",
                "rating": 5.0,
                "timestamp": "2026-06-12T20:00:00Z",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "accepted"

    def test_invalid_event_rejected(self, client):
        response = client.post(
            "/events",
            json={
                "user_id": 1,
                "movie_id": 318,
                "event_type": "rating",
                "rating": 9.0,  # out of range
                "timestamp": "2026-06-12T20:00:00Z",
            },
        )
        assert response.status_code == 422
