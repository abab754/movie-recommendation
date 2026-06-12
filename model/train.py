"""Train SVD collaborative filtering model on MovieLens data."""

import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "recommendations")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "rec_user")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "rec_pass")
MODEL_STORE_PATH = os.environ.get("MODEL_STORE_PATH", "/app/model_store")
RATINGS_FILE = os.environ.get("RATINGS_FILE", "/app/data/ml-1m/ratings.dat")

SVD_CONFIG = {
    "n_factors": 100,
    "n_epochs": 20,
    "lr_all": 0.005,
    "reg_all": 0.02,
}


def get_db_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def load_ratings_from_db():
    """Load ratings from the events table."""
    conn = get_db_connection()
    query = """
        SELECT user_id, movie_id, rating
        FROM events
        WHERE valid = TRUE AND event_type = 'rating' AND rating IS NOT NULL
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def load_ratings_from_file(filepath: str):
    """Load ratings from MovieLens 1M ratings.dat file (seed data)."""
    rows = []
    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) == 4:
                user_id, movie_id, rating, _ = parts
                rows.append(
                    {
                        "user_id": int(user_id),
                        "movie_id": int(movie_id),
                        "rating": float(rating),
                    }
                )
    return pd.DataFrame(rows)


def train_model(df: pd.DataFrame):
    """Train SVD model and return model, predictions, and RMSE."""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD(**SVD_CONFIG, random_state=42)
    model.fit(trainset)

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)

    return model, predictions, rmse


def save_model(model, version: str):
    """Save model artifact to disk."""
    os.makedirs(MODEL_STORE_PATH, exist_ok=True)
    path = os.path.join(MODEL_STORE_PATH, f"svd_{version}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    # Also save as 'latest' for easy loading
    latest_path = os.path.join(MODEL_STORE_PATH, "svd_latest.pkl")
    with open(latest_path, "wb") as f:
        pickle.dump(model, f)
    # Save version info
    version_path = os.path.join(MODEL_STORE_PATH, "version.txt")
    with open(version_path, "w") as f:
        f.write(version)
    print(f"Model saved: {path}")
    return path


def log_model_run(version: str, ndcg: float, hr: float, n_ratings: int, rmse: float):
    """Log training results to model_runs table."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_runs (version, ndcg_10, hr_10, n_ratings, rmse)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (version, float(ndcg), float(hr), int(n_ratings), float(rmse)),
            )
        conn.commit()
        conn.close()
        print(f"Logged model run: version={version}")
    except Exception as e:
        print(f"Failed to log model run: {e}")


def run_training():
    """Main training pipeline."""
    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    print(f"Starting training run: {version}")

    # Try loading from DB first, fall back to file
    try:
        df = load_ratings_from_db()
        if len(df) < 1000:
            raise ValueError(f"Only {len(df)} ratings in DB, using file instead")
        print(f"Loaded {len(df)} ratings from database")
    except Exception as e:
        print(f"DB load failed ({e}), loading from file...")
        df = load_ratings_from_file(RATINGS_FILE)
        print(f"Loaded {len(df)} ratings from file")

    print(f"Training SVD with config: {SVD_CONFIG}")
    model, predictions, rmse = train_model(df)

    # Compute evaluation metrics
    from model.evaluate import compute_ndcg_at_k, compute_hit_rate_at_k

    ndcg = compute_ndcg_at_k(predictions, k=10)
    hr = compute_hit_rate_at_k(predictions, k=10)

    print(f"Results: RMSE={rmse:.4f}, NDCG@10={ndcg:.4f}, HR@10={hr:.4f}")

    save_model(model, version)
    log_model_run(version, ndcg, hr, len(df), rmse)

    print("Training complete.")
    return model, version


if __name__ == "__main__":
    run_training()
