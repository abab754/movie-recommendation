"""Tests for model evaluation metrics (NDCG@10, HR@10)."""

from collections import namedtuple

import pytest

from model.evaluate import compute_hit_rate_at_k, compute_ndcg_at_k

# Mimics surprise's Prediction object
Prediction = namedtuple("Prediction", ["uid", "iid", "r_ui", "est", "details"])


def make_pred(uid, iid, true_rating, est_rating):
    return Prediction(uid, iid, true_rating, est_rating, {})


class TestHitRate:
    def test_perfect_hit_rate(self):
        # Every user has at least one relevant item (r_ui >= 3.5) in top-k
        preds = [
            make_pred("u1", "m1", 5.0, 4.8),
            make_pred("u1", "m2", 2.0, 3.0),
            make_pred("u2", "m3", 4.0, 4.5),
        ]
        assert compute_hit_rate_at_k(preds, k=10) == 1.0

    def test_zero_hit_rate(self):
        # No user has any relevant item
        preds = [
            make_pred("u1", "m1", 2.0, 4.8),
            make_pred("u2", "m2", 1.0, 4.5),
        ]
        assert compute_hit_rate_at_k(preds, k=10) == 0.0

    def test_partial_hit_rate(self):
        preds = [
            make_pred("u1", "m1", 5.0, 4.8),  # u1 has a hit
            make_pred("u2", "m2", 1.0, 4.5),  # u2 does not
        ]
        assert compute_hit_rate_at_k(preds, k=10) == 0.5

    def test_hit_outside_top_k_not_counted(self):
        # User has 3 items; relevant one has lowest predicted rating, k=2
        preds = [
            make_pred("u1", "m1", 2.0, 4.9),
            make_pred("u1", "m2", 2.0, 4.8),
            make_pred("u1", "m3", 5.0, 1.0),  # relevant but ranked 3rd
        ]
        assert compute_hit_rate_at_k(preds, k=2) == 0.0

    def test_empty_predictions(self):
        assert compute_hit_rate_at_k([], k=10) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        # Relevant items ranked highest -> NDCG = 1
        preds = [
            make_pred("u1", "m1", 5.0, 4.9),
            make_pred("u1", "m2", 4.0, 4.5),
            make_pred("u1", "m3", 1.0, 2.0),
        ]
        assert compute_ndcg_at_k(preds, k=10) == pytest.approx(1.0)

    def test_imperfect_ranking_lower_score(self):
        # Relevant item ranked below irrelevant ones -> NDCG < 1
        preds = [
            make_pred("u1", "m1", 1.0, 4.9),  # irrelevant, ranked 1st
            make_pred("u1", "m2", 1.0, 4.5),  # irrelevant, ranked 2nd
            make_pred("u1", "m3", 5.0, 2.0),  # relevant, ranked 3rd
        ]
        score = compute_ndcg_at_k(preds, k=10)
        assert 0 < score < 1.0

    def test_users_with_no_relevant_items_excluded(self):
        # u2 has no relevant items (idcg=0), shouldn't crash or skew
        preds = [
            make_pred("u1", "m1", 5.0, 4.9),
            make_pred("u2", "m2", 1.0, 4.5),
        ]
        assert compute_ndcg_at_k(preds, k=10) == pytest.approx(1.0)

    def test_empty_predictions(self):
        assert compute_ndcg_at_k([], k=10) == 0.0

    def test_ndcg_respects_k(self):
        # 3 relevant items but k=1: only the top one counts (still perfect order)
        preds = [
            make_pred("u1", f"m{i}", 5.0, 5.0 - i * 0.1) for i in range(3)
        ]
        score = compute_ndcg_at_k(preds, k=1)
        assert score == pytest.approx(1.0)
