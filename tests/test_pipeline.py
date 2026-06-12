"""Tests for the data pipeline: event schema validation and drift detection."""

import pytest
from pydantic import ValidationError

from pipeline.drift_detector import DriftDetector
from pipeline.schema import UserInteractionEvent


class TestEventSchema:
    def test_valid_rating_event(self):
        event = UserInteractionEvent(
            user_id=123,
            movie_id=456,
            event_type="rating",
            rating=4.5,
            timestamp="2026-01-01T12:00:00Z",
        )
        assert event.user_id == 123
        assert event.rating == 4.5

    def test_valid_click_event_without_rating(self):
        event = UserInteractionEvent(
            user_id=1,
            movie_id=2,
            event_type="click",
            timestamp="2026-01-01T12:00:00Z",
        )
        assert event.rating is None

    def test_invalid_event_type_rejected(self):
        with pytest.raises(ValidationError):
            UserInteractionEvent(
                user_id=1,
                movie_id=2,
                event_type="purchase",
                timestamp="2026-01-01T12:00:00Z",
            )

    def test_rating_too_high_rejected(self):
        with pytest.raises(ValidationError):
            UserInteractionEvent(
                user_id=1,
                movie_id=2,
                event_type="rating",
                rating=6.0,
                timestamp="2026-01-01T12:00:00Z",
            )

    def test_rating_too_low_rejected(self):
        with pytest.raises(ValidationError):
            UserInteractionEvent(
                user_id=1,
                movie_id=2,
                event_type="rating",
                rating=0.0,
                timestamp="2026-01-01T12:00:00Z",
            )

    def test_negative_user_id_rejected(self):
        with pytest.raises(ValidationError):
            UserInteractionEvent(
                user_id=-5,
                movie_id=2,
                event_type="rating",
                rating=3.0,
                timestamp="2026-01-01T12:00:00Z",
            )

    def test_missing_fields_rejected(self):
        with pytest.raises(ValidationError):
            UserInteractionEvent(user_id=1)


class TestDriftDetector:
    def _make_detector(self, baseline_size=100, window_size=50):
        detector = DriftDetector()
        detector.baseline_size = baseline_size
        detector.window_size = window_size
        return detector

    def test_baseline_not_set_initially(self):
        detector = self._make_detector()
        assert not detector.baseline_set

    def test_baseline_set_after_enough_ratings(self):
        detector = self._make_detector(baseline_size=100)
        for _ in range(100):
            detector.add_rating(3.5)
        assert detector.baseline_set
        assert detector.baseline_mean == pytest.approx(3.5)

    def test_no_drift_when_distribution_stable(self):
        detector = self._make_detector(baseline_size=100, window_size=50)
        # Baseline with variance: alternating 3 and 4
        for i in range(100):
            detector.add_rating(3.0 if i % 2 == 0 else 4.0)
        assert detector.baseline_set

        # Same distribution in window
        results = None
        for i in range(50):
            results = detector.add_rating(3.0 if i % 2 == 0 else 4.0)
        assert results is not None
        mean_result = next(r for r in results if r["metric"] == "rating_mean")
        assert not mean_result["drift_detected"]

    def test_drift_detected_when_mean_shifts(self):
        detector = self._make_detector(baseline_size=100, window_size=50)
        # Baseline around 3.5 with small variance
        for i in range(100):
            detector.add_rating(3.4 if i % 2 == 0 else 3.6)
        assert detector.baseline_set

        # Window of all 1.0 ratings — big shift
        results = None
        for _ in range(50):
            results = detector.add_rating(1.0)
        assert results is not None
        mean_result = next(r for r in results if r["metric"] == "rating_mean")
        assert mean_result["drift_detected"]

    def test_window_resets_after_check(self):
        detector = self._make_detector(baseline_size=10, window_size=5)
        for _ in range(10):
            detector.add_rating(3.0)
        for _ in range(5):
            detector.add_rating(3.0)
        assert len(detector.window) == 0

    def test_results_contain_both_metrics(self):
        detector = self._make_detector(baseline_size=10, window_size=5)
        for i in range(10):
            detector.add_rating(3.0 if i % 2 == 0 else 4.0)
        results = None
        for _ in range(5):
            results = detector.add_rating(3.5)
        metrics = {r["metric"] for r in results}
        assert metrics == {"rating_mean", "rating_stddev"}
