"""Loads SVD model and runs inference for recommendations."""

import os
import pickle

MODEL_STORE_PATH = os.environ.get("MODEL_STORE_PATH", "/app/model_store")

_model = None
_model_version = None


def load_model():
    """Load the latest SVD model into memory."""
    global _model, _model_version

    model_path = os.path.join(MODEL_STORE_PATH, "svd_latest.pkl")
    version_path = os.path.join(MODEL_STORE_PATH, "version.txt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")

    with open(model_path, "rb") as f:
        _model = pickle.load(f)

    if os.path.exists(version_path):
        with open(version_path, "r") as f:
            _model_version = f.read().strip()
    else:
        _model_version = "unknown"

    print(f"Loaded SVD model version: {_model_version}")


def get_model_version() -> str:
    return _model_version or "not_loaded"


def predict_ratings(user_id: int, movie_ids: list[int]) -> list[dict]:
    """Predict ratings for a user on a list of movies."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    predictions = []
    for movie_id in movie_ids:
        pred = _model.predict(user_id, movie_id)
        predictions.append(
            {
                "movie_id": movie_id,
                "predicted_rating": round(pred.est, 2),
            }
        )

    predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
    return predictions


def get_recommendations(user_id: int, all_movie_ids: list[int], rated_movie_ids: set[int], n: int = 10) -> list[dict]:
    """Get top-N recommendations for a user, excluding already-rated movies."""
    candidate_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]
    predictions = predict_ratings(user_id, candidate_ids)
    return predictions[:n]
