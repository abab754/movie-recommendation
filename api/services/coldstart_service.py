"""Popularity-based fallback for cold-start users."""

import os

import psycopg2

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "recommendations")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "rec_user")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "rec_pass")

# Precomputed popular movies: list of (movie_id, score)
_popular_movies: list[dict] = []


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


def precompute_popular_movies():
    """Compute top-100 popular movies by (rating_count * avg_rating)."""
    global _popular_movies

    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT movie_id,
                   COUNT(*) as rating_count,
                   AVG(rating) as avg_rating
            FROM events
            WHERE valid = TRUE AND event_type = 'rating' AND rating IS NOT NULL
            GROUP BY movie_id
            HAVING COUNT(*) >= 50
            ORDER BY COUNT(*) * AVG(rating) DESC
            LIMIT 100
            """
        )
        rows = cur.fetchall()
    conn.close()

    _popular_movies = [
        {
            "movie_id": row[0],
            "predicted_rating": round(float(row[2]), 2),
        }
        for row in rows
    ]
    print(f"Precomputed {len(_popular_movies)} popular movies for cold-start")


def get_recommendations(rated_movie_ids: set[int], n: int = 10) -> list[dict]:
    """Return top-N popular movies excluding ones the user already rated."""
    results = []
    for movie in _popular_movies:
        if movie["movie_id"] not in rated_movie_ids:
            results.append(movie)
        if len(results) >= n:
            break
    return results
