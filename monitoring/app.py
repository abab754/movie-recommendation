"""Streamlit monitoring dashboard for the recommendation platform."""

import os
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import requests
import streamlit as st

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "recommendations")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "rec_user")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "rec_pass")
API_URL = os.environ.get("API_URL", "http://api:8000")

st.set_page_config(page_title="Movie Rec Monitoring", layout="wide")
st.title("Movie Recommendation Platform — Monitoring")


@st.cache_resource
def get_connection():
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


def query_df(sql: str) -> pd.DataFrame:
    conn = get_connection()
    return pd.read_sql(sql, conn)


tab_health, tab_model, tab_drift, tab_ab, tab_demo = st.tabs(
    ["System Health", "Model Performance", "Data Drift", "A/B Testing", "Try It"]
)

# ---------------------------------------------------------------- Tab 1
with tab_health:
    st.header("System Health")

    col1, col2, col3, col4 = st.columns(4)

    # API health check
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        api_up = resp.status_code == 200
        model_version = resp.json().get("model_version", "unknown")
    except Exception:
        api_up = False
        model_version = "unreachable"

    col1.metric("API Status", "UP" if api_up else "DOWN")
    col2.metric("Model Version", model_version)

    # Latency metrics from API
    try:
        metrics = requests.get(f"{API_URL}/metrics", timeout=2).json()
        col3.metric("p50 Latency", f"{metrics['p50_ms']} ms")
        col4.metric("p95 Latency", f"{metrics['p95_ms']} ms")
    except Exception:
        col3.metric("p50 Latency", "n/a")
        col4.metric("p95 Latency", "n/a")

    # Request volume over time (from recommendations log)
    st.subheader("Recommendations Served Over Time")
    df_vol = query_df(
        """
        SELECT date_trunc('minute', served_at) AS minute, COUNT(*) AS requests
        FROM recommendations
        GROUP BY 1 ORDER BY 1
        """
    )
    if not df_vol.empty:
        st.line_chart(df_vol.set_index("minute")["requests"])
    else:
        st.info("No recommendations served yet.")

    # Latency distribution from logged recommendations
    st.subheader("Serving Latency (logged)")
    df_lat = query_df(
        """
        SELECT served_at, latency_ms FROM recommendations
        ORDER BY served_at DESC LIMIT 500
        """
    )
    if not df_lat.empty:
        st.line_chart(df_lat.set_index("served_at")["latency_ms"])
        pct_under_50 = (df_lat["latency_ms"] < 50).mean() * 100
        st.caption(f"{pct_under_50:.1f}% of requests under 50ms target")

# ---------------------------------------------------------------- Tab 2
with tab_model:
    st.header("Model Performance")

    df_runs = query_df(
        """
        SELECT version, trained_at, ndcg_10, hr_10, n_ratings, rmse
        FROM model_runs ORDER BY trained_at
        """
    )

    if df_runs.empty:
        st.info("No training runs logged yet.")
    else:
        latest = df_runs.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("NDCG@10", f"{latest['ndcg_10']:.4f}")
        col2.metric("HR@10", f"{latest['hr_10']:.4f}")
        col3.metric("RMSE", f"{latest['rmse']:.4f}")
        col4.metric("Ratings Used", f"{int(latest['n_ratings']):,}")

        st.caption(
            f"Last retrain: {latest['trained_at']} (version {latest['version']})"
        )

        st.subheader("Metrics Across Versions")
        chart_df = df_runs.set_index("trained_at")[["ndcg_10", "hr_10"]]
        st.line_chart(chart_df)

        st.subheader("RMSE Trend")
        st.line_chart(df_runs.set_index("trained_at")["rmse"])

        st.subheader("Training Data Growth")
        st.line_chart(df_runs.set_index("trained_at")["n_ratings"])

        st.dataframe(df_runs)

# ---------------------------------------------------------------- Tab 3
with tab_drift:
    st.header("Data Drift")

    df_drift = query_df(
        """
        SELECT logged_at, metric, baseline_value, current_value, drift_detected
        FROM drift_log ORDER BY logged_at
        """
    )

    if df_drift.empty:
        st.info(
            "No drift checks logged yet. The detector needs 100K ratings "
            "to establish its baseline before checks begin."
        )
    else:
        n_drift = int(df_drift["drift_detected"].sum())
        col1, col2 = st.columns(2)
        col1.metric("Total Drift Checks", len(df_drift))
        col2.metric("Drift Events Detected", n_drift)

        for metric_name in df_drift["metric"].unique():
            st.subheader(f"{metric_name} over time")
            sub = df_drift[df_drift["metric"] == metric_name]
            chart_df = sub.set_index("logged_at")[["baseline_value", "current_value"]]
            st.line_chart(chart_df)

        drifted = df_drift[df_drift["drift_detected"]]
        if not drifted.empty:
            st.subheader("Drift Events")
            st.dataframe(drifted.style.applymap(lambda _: "color: red"))

# ---------------------------------------------------------------- Tab 4
with tab_ab:
    st.header("A/B Testing: SVD vs Cold-Start")

    df_ab = query_df(
        """
        SELECT ab_variant,
               COUNT(*) AS recommendations_served,
               AVG(latency_ms) AS avg_latency_ms,
               COUNT(DISTINCT user_id) AS unique_users
        FROM recommendations
        WHERE ab_variant IS NOT NULL
        GROUP BY ab_variant
        """
    )

    if df_ab.empty:
        st.info("No A/B data yet — make some /recommend requests.")
    else:
        st.subheader("Variant Comparison")
        st.dataframe(df_ab)

        col1, col2 = st.columns(2)
        for _, row in df_ab.iterrows():
            target = col1 if row["ab_variant"] == "svd" else col2
            target.metric(
                f"{row['ab_variant']} — recommendations",
                f"{int(row['recommendations_served']):,}",
                f"{row['avg_latency_ms']:.1f} ms avg",
            )

        # Simulated CTR: fraction of recommended movies later rated by the user
        st.subheader("Recommendation Follow-up Rate by Variant")
        df_ctr = query_df(
            """
            SELECT r.ab_variant,
                   COUNT(DISTINCT e.id)::float / NULLIF(COUNT(DISTINCT r.id), 0) AS followup_rate
            FROM recommendations r
            LEFT JOIN events e
              ON e.user_id = r.user_id
             AND e.movie_id = ANY(r.movie_ids)
             AND e.timestamp > r.served_at
            WHERE r.ab_variant IS NOT NULL
            GROUP BY r.ab_variant
            """
        )
        st.dataframe(df_ctr)
        st.caption(
            "Follow-up rate = recommended movies the user later interacted with. "
            "Note: with replayed historical data this is a proxy metric, not true CTR. "
            "Statistical significance requires sufficient sample size per variant."
        )

# ---------------------------------------------------------------- Tab 5
with tab_demo:
    st.header("Try It — Live Recommendations")

    col1, col2 = st.columns(2)
    user_id = col1.number_input("User ID", min_value=1, value=1, step=1)
    variant_choice = col2.selectbox(
        "Variant", ["auto (A/B assignment)", "svd", "coldstart"]
    )

    if st.button("Get Recommendations"):
        headers = {}
        if variant_choice != "auto (A/B assignment)":
            headers["X-AB-Variant"] = variant_choice
        try:
            resp = requests.get(
                f"{API_URL}/recommend/{int(user_id)}", headers=headers, timeout=5
            )
            data = resp.json()

            m1, m2, m3 = st.columns(3)
            m1.metric("Variant Used", data["variant"])
            m2.metric("Model Version", data["model_version"])
            m3.metric("Latency", f"{data['latency_ms']} ms")

            recs_df = pd.DataFrame(data["recommendations"])
            recs_df.index = recs_df.index + 1  # rank from 1
            st.dataframe(
                recs_df[["title", "predicted_rating", "movie_id"]],
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Request failed: {e}")

st.sidebar.markdown(f"**Last refreshed:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
if st.sidebar.button("Refresh"):
    st.rerun()
