"""Listens for retrain triggers on Kafka and runs model training."""

import json
import os

from confluent_kafka import Consumer

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TRIGGER_TOPIC = os.environ.get("KAFKA_TRIGGER_TOPIC", "model-triggers")


def run_scheduler():
    print(f"Retrain scheduler listening on {KAFKA_TRIGGER_TOPIC}...")
    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "group.id": "retrain-scheduler-group",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([KAFKA_TRIGGER_TOPIC])

    try:
        while True:
            msg = consumer.poll(timeout=5.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            payload = json.loads(msg.value().decode("utf-8"))
            print(f"Received retrain trigger: {payload}")

            from model.train import run_training

            try:
                run_training()
            except Exception as e:
                print(f"Retraining failed: {e}")

    except KeyboardInterrupt:
        print("Scheduler shutting down...")
    finally:
        consumer.close()


if __name__ == "__main__":
    run_scheduler()
