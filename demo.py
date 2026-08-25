"""
CricketIQ — Quick Demo (pre-Step-6)
=====================================
Run: streamlit run demo.py

Shows real IPL data from the database using the analytics + visualization
layers built in Steps 3–5.  This is NOT the final UI (that's Step 6);
it's a smoke-check to confirm the full stack works end-to-end.

Stack exercised:
    PostgreSQL → db.queries → analytics.batting/bowling → visualization → Streamlit
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd

from src.db.connection import get_connection, test_connection
from src.db.queries import (
    get_database_summary,
    get_player_deliveries,
    search_players,
    get_all_players,
)
from src.analytics.batting import (
    get_batting_summary,
    get_phase_batting_stats,
    get_batting_season_trend,
    get_dismissal_breakdown,
)
from src.analytics.bowling import (
    get_bowling_summary,
    get_phase_bowling_stats,
)
from src.visualization.batting_charts import (
    phase_batting_chart,
    batting_season_trend_chart,
    dismissal_donut_chart,
    batting_summary_bars,
)
from src.visualization.bowling_charts import (
    phase_bowling_chart,
    bowling_season_trend_chart,
)
from src.visualization.style import apply_theme

apply_theme()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CricketIQ Demo",
    page_icon="🏏",
    layout="wide",
)

st.title("🏏 CricketIQ — Stack Demo")
st.caption("Pre-Step-6 smoke check · PostgreSQL → Analytics → Charts")

# ── DB health ─────────────────────────────────────────────────────────────────
st.header("📦 Database")
ok = test_connection()
if not ok:
    st.error("❌ Database connection failed — is PostgreSQL running?")
    st.stop()

summary = get_database_summary()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Matches",    f"{summary.get('total_matches', 0):,}")
c2.metric("Deliveries", f"{summary.get('total_deliveries', 0):,}")
c3.metric("Players",    f"{summary.get('total_players', 0):,}")
c4.metric("Teams",      f"{summary.get('total_teams', 0):,}")
c5.metric("Earliest ↔ Latest", f"{str(summary.get('earliest_match',''))[:4]} – {str(summary.get('latest_match',''))[:4]}")

st.divider()

# ── Player selector ──────────────────────────────────────────────────────────
st.header("🔍 Player Analytics")

@st.cache_data(ttl=300)
def load_players():
    return get_all_players(competition_id=None)

players_df = load_players()
if players_df.empty:
    st.warning("No players found in database.")
    st.stop()

# Sort and build a name → id map
players_df = players_df.sort_values("name")
name_to_id = dict(zip(players_df["name"], players_df["id"]))
player_names = players_df["name"].tolist()

# Default to Virat Kohli if present
default_idx = player_names.index("V Kohli") if "V Kohli" in player_names else 0

col_player, col_role = st.columns([3, 1])
with col_player:
    selected_name = st.selectbox("Select player", player_names, index=default_idx)
with col_role:
    role = st.radio("Role", ["batter", "bowler"], horizontal=True)

player_id = name_to_id[selected_name]

# ── Load deliveries ────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_deliveries(pid: int, r: str):
    return get_player_deliveries(player_id=pid, role=r)

with st.spinner(f"Loading {selected_name}'s deliveries..."):
    df = load_deliveries(player_id, role)

if df.empty:
    st.info(f"No {role} data found for {selected_name}.")
    st.stop()

st.success(f"Loaded **{len(df):,}** deliveries for **{selected_name}** as {role}.")

# ── Analytics ─────────────────────────────────────────────────────────────────
if role == "batter":
    summary    = get_batting_summary(df)
    phase_df   = get_phase_batting_stats(df)
    season_df  = get_batting_season_trend(df)
    dismiss_df = get_dismissal_breakdown(df)

    # Headline stats
    st.subheader(f"📊 {selected_name} — Batting Summary")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Runs",         summary.get("runs") or "—")
    m2.metric("Innings",      summary.get("innings") or "—")
    m3.metric("Average",      f"{summary.get('avg') or 0:.1f}" if summary.get('avg') else "∞")
    m4.metric("Strike Rate",  f"{summary.get('sr') or 0:.1f}" if summary.get('sr') else "—")
    m5.metric("50s / 100s",   f"{summary.get('fifties') or 0} / {summary.get('hundreds') or 0}")
    m6.metric("Best",         summary.get("highest_score") or "—")

    # Charts row 1
    ch1, ch2 = st.columns(2)
    with ch1:
        st.pyplot(phase_batting_chart(phase_df, selected_name), use_container_width=True)
    with ch2:
        st.pyplot(dismissal_donut_chart(dismiss_df, selected_name), use_container_width=True)

    # Charts row 2
    st.pyplot(batting_season_trend_chart(season_df, selected_name), use_container_width=True)

    # vs IPL Avg
    st.pyplot(batting_summary_bars(summary, selected_name), use_container_width=True)

else:  # bowler
    summary   = get_bowling_summary(df)
    phase_df  = get_phase_bowling_stats(df)
    season_df = get_bowling_season_trend(df)

    st.subheader(f"⚾ {selected_name} — Bowling Summary")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Wickets",    summary.get("wickets") or "—")
    m2.metric("Matches",    summary.get("matches") or "—")
    m3.metric("Economy",    f"{summary.get('economy') or 0:.2f}" if summary.get('economy') else "—")
    m4.metric("Average",    f"{summary.get('avg') or 0:.1f}" if summary.get('avg') else "—")
    m5.metric("SR",         f"{summary.get('sr') or 0:.1f}" if summary.get('sr') else "—")
    m6.metric("Best",       summary.get("best_bowling") or "—")

    ch1, ch2 = st.columns(2)
    with ch1:
        st.pyplot(phase_bowling_chart(phase_df, selected_name), use_container_width=True)
    with ch2:
        st.pyplot(bowling_season_trend_chart(season_df, selected_name), use_container_width=True)

# ── Raw phase data table ───────────────────────────────────────────────────────
with st.expander("📋 Raw phase data"):
    st.dataframe(phase_df, use_container_width=True)

st.divider()
st.caption("Step 6 will replace this with a full multi-page Streamlit app.")
