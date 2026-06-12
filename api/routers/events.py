"""POST /events endpoint for ingesting interaction events."""

import json
import os

from confluent_kafka import Producer
from fastapi import APIRouter, HTTPException

from pipeline.schema import UserInteractionEvent

router = APIRouter()

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_INTERACTIONS_TOPIC", "user-interactions")

_producer = None


def _get_producer():
    global _producer
    if _producer is None:
        _producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})
    return _producer


@router.post("/events")
def ingest_event(event: UserInteractionEvent):
    """Ingest a single user interaction event via Kafka."""
    producer = _get_producer()
    producer.produce(
        KAFKA_TOPIC,
        key=str(event.user_id),
        value=event.model_dump_json(),
    )
    producer.flush()
    return {"status": "accepted", "event": event.model_dump()}
