"""POST /events endpoint for ingesting interaction events."""

import os

from fastapi import APIRouter

from pipeline.schema import UserInteractionEvent

router = APIRouter()

KAFKA_ENABLED = os.environ.get("KAFKA_ENABLED", "true").lower() == "true"
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_INTERACTIONS_TOPIC", "user-interactions")

_producer = None


def _get_producer():
    global _producer
    if _producer is None:
        from confluent_kafka import Producer

        _producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})
    return _producer


def _insert_event_direct(event: UserInteractionEvent):
    """Direct-to-Postgres ingestion for deployments without Kafka."""
    from api.routers.recommend import get_db_connection

    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO events (user_id, movie_id, event_type, rating, timestamp, valid)
            VALUES (%s, %s, %s, %s, %s, TRUE)
            """,
            (
                event.user_id,
                event.movie_id,
                event.event_type,
                event.rating,
                event.timestamp,
            ),
        )
    conn.commit()
    conn.close()


@router.post("/events")
def ingest_event(event: UserInteractionEvent):
    """Ingest a single user interaction event."""
    if KAFKA_ENABLED:
        producer = _get_producer()
        producer.produce(
            KAFKA_TOPIC,
            key=str(event.user_id),
            value=event.model_dump_json(),
        )
        producer.flush()
    else:
        _insert_event_direct(event)
    return {"status": "accepted", "event": event.model_dump()}
