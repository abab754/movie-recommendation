"""Offline evaluation metrics: NDCG@10, HR@10."""

from collections import defaultdict

import numpy as np


def _get_top_k_per_user(predictions, k=10):
    """Group predictions by user and get top-k by estimated rating."""
    user_preds = defaultdict(list)
    for pred in predictions:
        user_preds[pred.uid].append(pred)

    top_k = {}
    for uid, preds in user_preds.items():
        preds.sort(key=lambda x: x.est, reverse=True)
        top_k[uid] = preds[:k]

    return top_k


def compute_ndcg_at_k(predictions, k=10, threshold=3.5):
    """Compute NDCG@K across all users.

    A recommendation is 'relevant' if the true rating >= threshold.
    """
    top_k = _get_top_k_per_user(predictions, k)

    ndcg_scores = []
    for uid, user_preds in top_k.items():
        # DCG: relevance at each position
        dcg = 0.0
        for i, pred in enumerate(user_preds):
            rel = 1.0 if pred.r_ui >= threshold else 0.0
            dcg += rel / np.log2(i + 2)  # i+2 because positions start at 1

        # Ideal DCG: sort by actual relevance
        ideal_rels = sorted(
            [1.0 if p.r_ui >= threshold else 0.0 for p in user_preds], reverse=True
        )
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def compute_hit_rate_at_k(predictions, k=10, threshold=3.5):
    """Compute Hit Rate@K: fraction of users with at least one relevant
    item in their top-K recommendations."""
    top_k = _get_top_k_per_user(predictions, k)

    hits = 0
    for uid, user_preds in top_k.items():
        if any(pred.r_ui >= threshold for pred in user_preds):
            hits += 1

    return hits / len(top_k) if top_k else 0.0
