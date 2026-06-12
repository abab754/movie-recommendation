"""Tests for SVD, cold-start service logic, and producer parsing."""

import pytest

from api.services import coldstart_service, svd_service
from pipeline.producer import parse_ratings_file


class FakeSVDModel:
    """Mimics surprise SVD: predict(uid, iid) -> object with .est"""

    class Pred:
        def __init__(self, est):
            self.est = est

    def predict(self, user_id, movie_id):
        # Deterministic fake: higher movie_id -> higher predicted rating
        return self.Pred(est=min(5.0, 1.0 + movie_id * 0.5))


class TestSVDService:
    @pytest.fixture(autouse=True)
    def fake_model(self, monkeypatch):
        monkeypatch.setattr(svd_service, "_model", FakeSVDModel())
        monkeypatch.setattr(svd_service, "_model_version", "v_test")

    def test_predict_ratings_sorted_descending(self):
        preds = svd_service.predict_ratings(1, [1, 2, 3])
        ratings = [p["predicted_rating"] for p in preds]
        assert ratings == sorted(ratings, reverse=True)

    def test_get_recommendations_excludes_rated(self):
        recs = svd_service.get_recommendations(
            1, all_movie_ids=[1, 2, 3, 4], rated_movie_ids={3, 4}, n=10
        )
        rec_ids = {r["movie_id"] for r in recs}
        assert rec_ids == {1, 2}

    def test_get_recommendations_respects_n(self):
        recs = svd_service.get_recommendations(
            1, all_movie_ids=list(range(1, 21)), rated_movie_ids=set(), n=5
        )
        assert len(recs) == 5

    def test_model_version(self):
        assert svd_service.get_model_version() == "v_test"

    def test_predict_without_model_raises(self, monkeypatch):
        monkeypatch.setattr(svd_service, "_model", None)
        with pytest.raises(RuntimeError):
            svd_service.predict_ratings(1, [1])


class TestColdstartService:
    @pytest.fixture(autouse=True)
    def fake_popular(self, monkeypatch):
        monkeypatch.setattr(
            coldstart_service,
            "_popular_movies",
            [{"movie_id": i, "predicted_rating": 4.0} for i in range(1, 21)],
        )

    def test_returns_top_n(self):
        recs = coldstart_service.get_recommendations(set(), n=10)
        assert len(recs) == 10
        assert recs[0]["movie_id"] == 1

    def test_excludes_rated_movies(self):
        recs = coldstart_service.get_recommendations({1, 2, 3}, n=5)
        rec_ids = {r["movie_id"] for r in recs}
        assert rec_ids == {4, 5, 6, 7, 8}

    def test_fewer_than_n_available(self):
        rated = set(range(1, 16))  # 15 of 20 rated
        recs = coldstart_service.get_recommendations(rated, n=10)
        assert len(recs) == 5


class TestProducerParsing:
    def test_parse_ratings_file(self, tmp_path):
        ratings_file = tmp_path / "ratings.dat"
        ratings_file.write_text(
            "1::1193::5::978300760\n"
            "1::661::3::978302109\n"
            "malformed line\n"
            "2::1357::4::978298709\n"
        )
        events = list(parse_ratings_file(str(ratings_file)))
        assert len(events) == 3  # malformed line skipped
        assert events[0]["user_id"] == 1
        assert events[0]["movie_id"] == 1193
        assert events[0]["rating"] == 5.0
        assert events[0]["event_type"] == "rating"
        assert "timestamp" in events[0]
