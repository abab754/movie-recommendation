"""Tests for consumer DB helpers and training pipeline helpers."""

import json
import os
import pickle

import pytest

from model import train
from pipeline import consumer


class FakeCursor:
    def __init__(self):
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeConnection:
    def __init__(self):
        self.cur = FakeCursor()
        self.committed = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True

    def close(self):
        pass


class FakeKafkaProducer:
    def __init__(self):
        self.messages = []

    def produce(self, topic, value=None, key=None, callback=None):
        self.messages.append((topic, value))

    def flush(self):
        pass


class TestConsumerHelpers:
    def test_insert_event_valid(self):
        conn = FakeConnection()
        event = {
            "user_id": 1,
            "movie_id": 2,
            "event_type": "rating",
            "rating": 4.0,
            "timestamp": "2026-01-01T00:00:00Z",
        }
        consumer.insert_event(conn, event, valid=True)
        assert conn.committed
        query, params = conn.cur.executed[0]
        assert "INSERT INTO events" in query
        assert params[0] == 1
        assert params[5] is True

    def test_insert_event_invalid_flag(self):
        conn = FakeConnection()
        consumer.insert_event(conn, {"user_id": 9}, valid=False)
        _, params = conn.cur.executed[0]
        assert params[5] is False

    def test_insert_drift_log(self):
        conn = FakeConnection()
        results = [
            {
                "metric": "rating_mean",
                "baseline_value": 3.5,
                "current_value": 1.2,
                "drift_detected": True,
            },
            {
                "metric": "rating_stddev",
                "baseline_value": 1.0,
                "current_value": 1.1,
                "drift_detected": False,
            },
        ]
        consumer.insert_drift_log(conn, results)
        assert len(conn.cur.executed) == 2
        assert conn.committed

    def test_send_retrain_trigger(self):
        producer = FakeKafkaProducer()
        consumer.send_retrain_trigger(producer, 10000)
        assert len(producer.messages) == 1
        topic, value = producer.messages[0]
        payload = json.loads(value)
        assert payload["trigger"] == "retrain"
        assert payload["valid_ratings_count"] == 10000


class TestTrainHelpers:
    def test_load_ratings_from_file(self, tmp_path):
        ratings_file = tmp_path / "ratings.dat"
        ratings_file.write_text(
            "1::1193::5::978300760\n"
            "2::661::3::978302109\n"
        )
        df = train.load_ratings_from_file(str(ratings_file))
        assert len(df) == 2
        assert list(df.columns) == ["user_id", "movie_id", "rating"]
        assert df.iloc[0]["rating"] == 5.0

    def test_save_model_writes_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(train, "MODEL_STORE_PATH", str(tmp_path))
        fake_model = {"weights": [1, 2, 3]}

        path = train.save_model(fake_model, "v_test")

        assert os.path.exists(path)
        assert os.path.exists(tmp_path / "svd_latest.pkl")
        with open(tmp_path / "version.txt") as f:
            assert f.read() == "v_test"
        with open(tmp_path / "svd_latest.pkl", "rb") as f:
            assert pickle.load(f) == fake_model

    def test_train_model_on_small_dataset(self):
        import pandas as pd

        # Tiny synthetic dataset: 20 users x 10 movies
        rows = []
        for u in range(1, 21):
            for m in range(1, 11):
                rows.append(
                    {"user_id": u, "movie_id": m, "rating": float((u + m) % 5 + 1)}
                )
        df = pd.DataFrame(rows)

        model, predictions, rmse = train.train_model(df)

        assert rmse > 0
        assert len(predictions) > 0
        # Model can make a prediction for any user/movie pair
        pred = model.predict(1, 1)
        assert 1.0 <= pred.est <= 5.0

    def test_log_model_run(self, monkeypatch):
        conn = FakeConnection()
        monkeypatch.setattr(train, "get_db_connection", lambda: conn)

        train.log_model_run("v1", 0.5, 0.7, 1000, 0.9)

        assert conn.committed
        query, params = conn.cur.executed[0]
        assert "INSERT INTO model_runs" in query
        assert params == ("v1", 0.5, 0.7, 1000, 0.9)
