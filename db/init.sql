-- Events table: raw user interactions from Kafka
CREATE TABLE IF NOT EXISTS events (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    movie_id    INTEGER NOT NULL,
    event_type  VARCHAR(20) NOT NULL,
    rating      FLOAT,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ab_variant  VARCHAR(10),
    valid       BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_events_user_id ON events(user_id);
CREATE INDEX IF NOT EXISTS idx_events_movie_id ON events(movie_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);

-- Recommendations table: logged recommendations served
CREATE TABLE IF NOT EXISTS recommendations (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL,
    movie_ids       INTEGER[] NOT NULL,
    served_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    latency_ms      FLOAT,
    ab_variant      VARCHAR(10),
    model_version   VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_recommendations_user_id ON recommendations(user_id);

-- Model runs table: training history
CREATE TABLE IF NOT EXISTS model_runs (
    id          SERIAL PRIMARY KEY,
    version     VARCHAR(50) NOT NULL,
    trained_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ndcg_10     FLOAT,
    hr_10       FLOAT,
    n_ratings   INTEGER,
    rmse        FLOAT
);

-- Drift log table: data drift tracking
CREATE TABLE IF NOT EXISTS drift_log (
    id              SERIAL PRIMARY KEY,
    logged_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metric          VARCHAR(50),
    baseline_value  FLOAT,
    current_value   FLOAT,
    drift_detected  BOOLEAN
);
