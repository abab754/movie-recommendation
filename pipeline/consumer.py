"""Kafka consumer that reads user interaction events, validates them,
stores to Postgres, and triggers retraining + drift detection."""

import json
import os
import time

import psycopg2
from confluent_kafka import Consumer, Producer

from pipeline.drift_detector import DriftDetector
from pipeline.schema import UserInteractionEvent

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_INTERACTIONS_TOPIC = os.environ.get("KAFKA_INTERACTIONS_TOPIC", "user-interactions")
KAFKA_TRIGGER_TOPIC = os.environ.get("KAFKA_TRIGGER_TOPIC", "model-triggers")
RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", "10000"))

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "recommendations")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "rec_user")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "rec_pass")


def get_db_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def insert_event(conn, event_data: dict, valid: bool):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO events (user_id, movie_id, event_type, rating, timestamp, valid)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                event_data.get("user_id"),
                event_data.get("movie_id"),
                event_data.get("event_type"),
                event_data.get("rating"),
                event_data.get("timestamp"),
                valid,
            ),
        )
    conn.commit()


def insert_drift_log(conn, drift_results: list[dict]):
    with conn.cursor() as cur:
        for result in drift_results:
            cur.execute(
                """
                INSERT INTO drift_log (metric, baseline_value, current_value, drift_detected)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    result["metric"],
                    result["baseline_value"],
                    result["current_value"],
                    result["drift_detected"],
                ),
            )
    conn.commit()


def send_retrain_trigger(producer: Producer, valid_count: int):
    msg = json.dumps({"trigger": "retrain", "valid_ratings_count": valid_count})
    producer.produce(KAFKA_TRIGGER_TOPIC, value=msg)
    producer.flush()
    print(f"Sent retrain trigger at {valid_count} valid ratings")


def wait_for_postgres(max_retries=30, delay=2):
    """Wait for Postgres to be ready before proceeding."""
    for attempt in range(1, max_retries + 1):
        try:
            conn = get_db_connection()
            conn.close()
            print("Postgres is ready.")
            return
        except psycopg2.OperationalError:
            print(f"Waiting for Postgres... (attempt {attempt}/{max_retries})")
            time.sleep(delay)
    raise RuntimeError("Could not connect to Postgres after max retries")


def run_consumer():
    # Wait for dependencies to be ready
    wait_for_postgres()

    print(f"Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}...")
    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "group.id": "event-consumer-group",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([KAFKA_INTERACTIONS_TOPIC])

    trigger_producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})

    print(f"Connecting to Postgres at {POSTGRES_HOST}:{POSTGRES_PORT}...")
    conn = get_db_connection()

    drift_detector = DriftDetector()
    valid_count = 0
    total_count = 0
    last_retrain_at = 0

    print("Consumer started. Waiting for messages...")
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            raw_value = msg.value().decode("utf-8")
            total_count += 1

            try:
                event_data = json.loads(raw_value)
                # Validate with Pydantic
                UserInteractionEvent(**event_data)
                valid = True
            except Exception as e:
                valid = False
                if total_count % 1000 == 0:
                    print(f"Validation error (event #{total_count}): {e}")

            # Store event (valid or invalid)
            try:
                insert_event(conn, event_data if valid else json.loads(raw_value), valid)
            except Exception as e:
                print(f"DB insert error: {e}")
                conn = get_db_connection()
                continue

            if valid:
                valid_count += 1

                # Drift detection for rating events
                rating = event_data.get("rating")
                if rating is not None:
                    drift_results = drift_detector.add_rating(rating)
                    if drift_results:
                        try:
                            insert_drift_log(conn, drift_results)
                        except Exception as e:
                            print(f"Drift log insert error: {e}")

                # Retrain trigger every RETRAIN_THRESHOLD valid ratings
                if (
                    valid_count - last_retrain_at >= RETRAIN_THRESHOLD
                    and valid_count > 0
                ):
                    send_retrain_trigger(trigger_producer, valid_count)
                    last_retrain_at = valid_count

            if total_count % 5000 == 0:
                print(
                    f"Processed {total_count} events "
                    f"({valid_count} valid, {total_count - valid_count} invalid)"
                )

    except KeyboardInterrupt:
        print("Consumer shutting down...")
    finally:
        consumer.close()
        conn.close()


if __name__ == "__main__":
    run_consumer()
