"""Kafka producer that replays MovieLens 1M ratings as streaming events."""

import json
import os
import time
from datetime import datetime, timezone

from confluent_kafka import Producer

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_INTERACTIONS_TOPIC", "user-interactions")
PRODUCER_DELAY_MS = int(os.environ.get("PRODUCER_DELAY_MS", "100"))
RATINGS_FILE = os.environ.get("RATINGS_FILE", "/app/data/ml-1m/ratings.dat")


def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed: {err}")


def parse_ratings_file(filepath: str):
    """Parse MovieLens 1M ratings.dat (UserID::MovieID::Rating::Timestamp)."""
    with open(filepath, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue
            user_id, movie_id, rating, ts = parts
            yield {
                "user_id": int(user_id),
                "movie_id": int(movie_id),
                "event_type": "rating",
                "rating": float(rating),
                "timestamp": datetime.fromtimestamp(
                    int(ts), tz=timezone.utc
                ).isoformat(),
            }


def run_producer():
    print(f"Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}...")
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})

    delay_sec = PRODUCER_DELAY_MS / 1000.0
    count = 0

    print(f"Reading ratings from {RATINGS_FILE}...")
    for event in parse_ratings_file(RATINGS_FILE):
        producer.produce(
            KAFKA_TOPIC,
            key=str(event["user_id"]),
            value=json.dumps(event),
            callback=delivery_report,
        )
        count += 1
        if count % 1000 == 0:
            producer.flush()
            print(f"Produced {count} events")

        time.sleep(delay_sec)

    producer.flush()
    print(f"Finished producing {count} events.")


if __name__ == "__main__":
    run_producer()
