# ui.py
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import math
import datetime

st.set_page_config(page_title="Sentinel AI | Financial Compliance", layout="wide")

# --- Config ---
CSV_FILE = "suspicious_alerts.csv"
DOC_PDF = "./data/documents/PS WEEK 1.pdf"  # fixed path

# --- Header / top bar ---
st.markdown(
    """
    <style>
    .metric-card {background-color: #0f1720; padding: 14px; border-radius: 8px; border-left: 4px solid #ff6b6b;}
    .small-muted {color: #9aa3ad; font-size:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

col_title, col_doc = st.columns([5, 1])
with col_title:
    st.title("🛡️ Sentinel AI: Real-Time Compliance Engine")
    st.markdown("Monitoring live transaction streams with Pathway Temporal Windows + RAG.")
with col_doc:
    p = Path(DOC_PDF)
    st.markdown(
        f"[📘 Problem PDF]({p.as_posix()})" if p.exists() else "📘 PDF not found",
        unsafe_allow_html=True,
    )

st.markdown("---")

# --- helper functions ---
def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
    except Exception:
        try:
            df = pd.read_csv(p, encoding="utf-8", errors="replace")
        except Exception:
            return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]
    col_map = {}

    # handle column name variations
    if "latest_time" in df.columns:
        col_map["time"] = "latest_time"
    elif "time" in df.columns:
        col_map["time"] = "time"

    if "amount_f" in df.columns:
        col_map["amount"] = "amount_f"
    elif "amount" in df.columns:
        col_map["amount"] = "amount"

    # FIX: map reduced column correctly
    if "velocity_avg_1h" in df.columns:
        col_map["velocity_avg_1h"] = "velocity_avg_1h"
    elif "velocity_sum_1h" in df.columns:
        col_map["velocity_avg_1h"] = "velocity_sum_1h"

    # apply renames
    rename_map = {}
    if "time" in col_map:
        rename_map[col_map["time"]] = "time"
    if "amount" in col_map:
        rename_map[col_map["amount"]] = "amount"
    if "velocity_avg_1h" in col_map:
        rename_map[col_map["velocity_avg_1h"]] = "velocity_avg_1h"

    if rename_map:
        df = df.rename(columns=rename_map)

    # numeric coercion
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    if "velocity_avg_1h" in df.columns:
        df["velocity_avg_1h"] = pd.to_numeric(df["velocity_avg_1h"], errors="coerce").fillna(0.0)

    # timestamp parsing
    if "time" in df.columns:
        try:
            df["time_parsed"] = pd.to_datetime(
                df["time"], format="%Y-%m-%dT%H:%M:%S", errors="coerce"
            )
        except Exception:
            df["time_parsed"] = pd.NaT
    else:
        df["time_parsed"] = pd.NaT

    return df


# --- refresh mechanism ---
try:
    from streamlit_autorefresh import st_autorefresh

    refresh_count = st_autorefresh(interval=2000, limit=None, key="autorefresh")
    REFRESH_MODE = "auto"
except Exception:
    REFRESH_MODE = "manual"
    st.markdown("Auto-refresh library not installed. Use the **Refresh** button below.")
    if st.button("🔄 Refresh now"):
        st.experimental_rerun()

# --- layout containers ---
left, right = st.columns([3, 2])

with left:
    df = load_csv(CSV_FILE)
    if df.empty:
        st.info("⏳ Waiting for transaction stream ingestion or CSV creation.")
    else:
        # metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total AI Alerts Triggered", len(df))
        m2.metric("High-Value Alerts (>$5k)", int((df["amount"] > 5000).sum()))
        m3.metric(
            "Watchlist Hits",
            int(df["watchlist_risk"].notna().sum()) if "watchlist_risk" in df.columns else 0,
        )

        latest_velocity = (
            df["velocity_avg_1h"].iloc[-1]
            if "velocity_avg_1h" in df.columns and not df["velocity_avg_1h"].isna().all()
            else 0.0
        )
        m4.metric("Latest Velocity (1h)", f"${latest_velocity:,.2f}")

        st.markdown("### 🔎 Transaction Velocity Over Time")

        chart_df = df.copy()

        # ensure we have some time axis
        if "time_parsed" not in chart_df or chart_df["time_parsed"].isna().all():
            chart_df["time_parsed"] = [
                datetime.datetime.now() - datetime.timedelta(seconds=5 * (len(chart_df) - i))
                for i in range(len(chart_df))
            ]

        # sort by time and keep only recent history
        chart_df = chart_df.sort_values("time_parsed").tail(200)

        # relative time in seconds since first alert (for a tight real-time axis)
        if chart_df["time_parsed"].notna().any():
            t0 = chart_df["time_parsed"].min()
            chart_df["elapsed_sec"] = (
                chart_df["time_parsed"] - t0
            ).dt.total_seconds()
        else:
            chart_df["elapsed_sec"] = list(range(len(chart_df)))

        # fallback if missing column
        if "velocity_avg_1h" not in chart_df:
            chart_df["velocity_avg_1h"] = chart_df.get("amount", 0.0)

        base = alt.Chart(chart_df).encode(
            x=alt.X("elapsed_sec:Q", title="Time (seconds, recent alerts)")
        )

        points = base.mark_circle(size=60).encode(
            y=alt.Y("velocity_avg_1h:Q", title="Velocity Avg (1h)"),
            color=alt.Color("watchlist_risk:N", title="Watchlist"),
            tooltip=[
                alt.Tooltip("time_parsed:T", title="Event time"),
                alt.Tooltip("elapsed_sec:Q", title="Seconds since start"),
                alt.Tooltip("user_id:N"),
                alt.Tooltip("amount:Q"),
                alt.Tooltip("velocity_avg_1h:Q"),
                alt.Tooltip("watchlist_risk:N"),
                alt.Tooltip("analysis:N"),
            ],
        )

        line = base.mark_line().encode(y="velocity_avg_1h:Q")

        chart = (line + points).interactive().properties(height=320, width="container")
        st.altair_chart(chart, use_container_width=True)

        st.markdown("### 🔴 Live Feed Analysis")

        preferred = ["time", "user_id", "amount", "velocity_avg_1h", "watchlist_risk", "analysis"]
        display_cols = [c for c in preferred if c in df.columns]
        if not display_cols:
            display_cols = df.columns.tolist()

        st.dataframe(df[display_cols].tail(200), use_container_width=True, height=300)

with right:
    st.markdown("### ⚙️ Controls & Diagnostics")
    st.markdown(f"- CSV file: `{CSV_FILE}`")
    st.markdown(f"- Document: `{Path(DOC_PDF).name}`")
    st.markdown(f"- Auto-refresh mode: **{REFRESH_MODE}**")

    if df is not None and not df.empty:
        st.markdown("#### Top flagged accounts (by velocity avg)")
        top = (
            df.groupby("user_id")
            .agg({"velocity_avg_1h": "max", "amount": "max"})
            .reset_index()
            .sort_values("velocity_avg_1h", ascending=False)
            .head(10)
        )
        st.table(top)

    st.markdown("---")
    st.markdown(
        """
        - Dashboard reads from `suspicious_alerts.csv`.
        - Auto-refresh every 2 seconds if library is installed.
        - UI uses fallback logic if expected columns are missing.
    """
    )
