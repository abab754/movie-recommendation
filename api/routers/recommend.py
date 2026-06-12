"""GET /recommend/{user_id} endpoint."""

import os
import time
from typing import Optional

import psycopg2
from fastapi import APIRouter, Header

from api.services import ab_service, coldstart_service, svd_service

router = APIRouter()

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "recommendations")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "rec_user")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "rec_pass")

# Cache of all movie IDs and titles (loaded at startup)
_all_movie_ids: list[int] = []
_movie_titles: dict[int, str] = {}


def get_db_connection():
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return psycopg2.connect(database_url)
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def load_movie_metadata():
    """Load movie titles from MovieLens movies.dat file."""
    global _all_movie_ids, _movie_titles
    movies_file = os.environ.get("MOVIES_FILE", "/app/data/ml-1m/movies.dat")
    if not os.path.exists(movies_file):
        print(f"Movies file not found: {movies_file}")
        return

    with open(movies_file, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) >= 2:
                movie_id = int(parts[0])
                title = parts[1]
                _all_movie_ids.append(movie_id)
                _movie_titles[movie_id] = title

    print(f"Loaded {len(_all_movie_ids)} movie titles")


def get_user_rated_movies(user_id: int) -> set[int]:
    """Get set of movie IDs the user has rated."""
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT movie_id FROM events WHERE user_id = %s AND valid = TRUE",
            (user_id,),
        )
        movie_ids = {row[0] for row in cur.fetchall()}
    conn.close()
    return movie_ids


def log_recommendation(user_id: int, movie_ids: list[int], latency_ms: float, variant: str, model_version: str):
    """Log served recommendation to the database."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO recommendations (user_id, movie_ids, latency_ms, ab_variant, model_version)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (user_id, movie_ids, latency_ms, variant, model_version),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to log recommendation: {e}")


@router.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10, x_ab_variant: Optional[str] = Header(None)):
    """Get top-N movie recommendations for a user."""
    start = time.perf_counter()

    rated_movie_ids = get_user_rated_movies(user_id)
    has_enough_ratings = len(rated_movie_ids) >= 5

    # Determine variant
    if x_ab_variant:
        variant = x_ab_variant
    elif has_enough_ratings:
        variant = ab_service.get_variant(user_id)
    else:
        variant = "coldstart"

    # Get recommendations based on variant
    if variant == "svd" and has_enough_ratings:
        recs = svd_service.get_recommendations(
            user_id, _all_movie_ids, rated_movie_ids, n=n
        )
        model_version = svd_service.get_model_version()
    else:
        variant = "coldstart"
        recs = coldstart_service.get_recommendations(rated_movie_ids, n=n)
        model_version = "coldstart"

    # Add titles
    for rec in recs:
        rec["title"] = _movie_titles.get(rec["movie_id"], "Unknown")

    latency_ms = (time.perf_counter() - start) * 1000

    # Log recommendation
    rec_movie_ids = [r["movie_id"] for r in recs]
    log_recommendation(user_id, rec_movie_ids, latency_ms, variant, model_version)

    return {
        "user_id": user_id,
        "variant": variant,
        "model_version": model_version,
        "recommendations": recs,
        "latency_ms": round(latency_ms, 1),
    }
