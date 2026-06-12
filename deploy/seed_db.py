"""One-time seed script for the deployed (hosted) Postgres.

Creates the schema and loads a subset of MovieLens ratings into the events
table. Run locally against the hosted database:

    DATABASE_URL=postgresql://user:pass@host:port/db python deploy/seed_db.py
"""

import io
import os
import sys
from datetime import datetime, timezone

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
RATINGS_FILE = os.environ.get(
    "RATINGS_FILE",
    os.path.join(os.path.dirname(__file__), "..", "data", "ml-1m", "ratings.dat"),
)
SEED_LIMIT = int(os.environ.get("SEED_LIMIT", "200000"))
INIT_SQL = os.path.join(os.path.dirname(__file__), "..", "db", "init.sql")


def main():
    if not DATABASE_URL:
        sys.exit("Set DATABASE_URL to the hosted Postgres connection string.")

    conn = psycopg2.connect(DATABASE_URL)

    # Create schema
    with open(INIT_SQL) as f:
        with conn.cursor() as cur:
            cur.execute(f.read())
    conn.commit()
    print("Schema created.")

    # Skip if already seeded
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM events")
        count = cur.fetchone()[0]
    if count > 0:
        sys.exit(f"events table already has {count} rows — skipping seed.")

    # Build a COPY buffer from ratings.dat (much faster than INSERTs)
    print(f"Loading up to {SEED_LIMIT} ratings from {RATINGS_FILE}...")
    buffer = io.StringIO()
    n = 0
    with open(RATINGS_FILE, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue
            user_id, movie_id, rating, ts = parts
            timestamp = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
            buffer.write(f"{user_id}\t{movie_id}\trating\t{rating}\t{timestamp}\tt\n")
            n += 1
            if n >= SEED_LIMIT:
                break

    buffer.seek(0)
    with conn.cursor() as cur:
        cur.copy_from(
            buffer,
            "events",
            columns=("user_id", "movie_id", "event_type", "rating", "timestamp", "valid"),
        )
    conn.commit()
    print(f"Seeded {n} events.")
    conn.close()


if __name__ == "__main__":
    main()
