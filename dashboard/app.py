"""
Streamlit dashboard for viewing eval run history and trends.
Run with: streamlit run dashboard/app.py
Free to host on Streamlit Community Cloud.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from src.db import get_recent_runs

st.set_page_config(page_title="Model Regression Dashboard", layout="wide")
st.title("🔍 Model Regression Detection Dashboard")

runs = get_recent_runs(limit=50)

if not runs:
    st.warning("No eval runs found yet. Run `python main.py --prompt-version v1` first.")
    st.stop()

# Reverse to chronological order for the trend chart
runs_chrono = list(reversed(runs))

df = pd.DataFrame([
    {
        "run_id": r["run_id"],
        "timestamp": r["timestamp"],
        "prompt_version": r["prompt_version"],
        "pass_rate": r["aggregate_stats"]["category_pass_rate"],
        "avg_summary_score": r["aggregate_stats"]["avg_summary_score"],
        "avg_latency_ms": r["aggregate_stats"]["avg_latency_ms"],
        "error_count": r["aggregate_stats"]["error_count"],
    }
    for r in runs_chrono
])

col1, col2, col3, col4 = st.columns(4)
latest = df.iloc[-1]
col1.metric("Latest Pass Rate", f"{latest['pass_rate']:.1%}")
col2.metric("Latest Prompt Version", latest["prompt_version"])
col3.metric("Avg Summary Score", f"{latest['avg_summary_score']}/5")
col4.metric("Avg Latency", f"{latest['avg_latency_ms']:.0f}ms")

st.subheader("Pass Rate Over Time")
st.line_chart(df.set_index("timestamp")["pass_rate"])

st.subheader("Summary Quality Over Time")
st.line_chart(df.set_index("timestamp")["avg_summary_score"])

st.subheader("Latency Over Time")
st.line_chart(df.set_index("timestamp")["avg_latency_ms"])

st.subheader("Run History")
st.dataframe(
    df[["run_id", "timestamp", "prompt_version", "pass_rate", "avg_summary_score", "error_count"]]
    .sort_values("timestamp", ascending=False),
    use_container_width=True,
)

st.subheader("Inspect a Run")
selected_run_id = st.selectbox("Select run", df["run_id"].tolist()[::-1])
selected_run = next(r for r in runs if r["run_id"] == selected_run_id)

results_df = pd.DataFrame(selected_run["results"])
st.dataframe(
    results_df[["test_case_id", "expected_category", "actual_category", "category_match", "summary_score", "latency_ms", "difficulty"]],
    use_container_width=True,
)
