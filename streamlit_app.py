"""
CricketIQ — Home Page
=======================
Entry point for the Streamlit multi-page app.
Run: streamlit run streamlit_app.py

Navigation to other pages is automatic via the pages/ folder:
  - Player Intelligence
  - Matchup Engine
  - Team Intelligence

DEPLOYMENT NOTE — Streamlit secrets bridge:
    Streamlit Community Cloud stores secrets in the cloud UI (not in .env).
    Pydantic-settings reads from env vars, NOT from st.secrets directly.
    The block below copies Streamlit secrets → os.environ BEFORE any
    src.* imports so pydantic-settings picks them up on first load.
    On local dev the .env file takes precedence (loaded by pydantic-settings).
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# ── Streamlit secrets → env vars bridge (cloud deployment) ───────────────────
# Must happen before any src.* import that triggers settings initialisation.
try:
    import streamlit as st
    _SECRETS_KEYS = ["DATABASE_URL", "LLM_API_KEY", "ENVIRONMENT", "LOG_LEVEL"]
    for _key in _SECRETS_KEYS:
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass   # Not running in Streamlit Cloud — ignore silently

import matplotlib
matplotlib.use("Agg")

import streamlit as st
import pandas as pd

from app.utils import (
    cached_db_ok,
    cached_db_summary,
    cached_matches,
    cached_players,
    cached_seasons,
)
from app.components import require_db, format_result
from src.visualization.style import apply_theme

apply_theme()

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="CricketIQ",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar branding ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏏 CricketIQ")
    st.markdown("*AI-powered IPL analytics*")
    st.divider()
    st.markdown(
        "**Navigate using the pages above** ↑\n\n"
        "- 🏏 Player Intelligence\n"
        "- ⚔️ Matchup Engine\n"
        "- 🏆 Team Intelligence"
    )
    st.divider()
    st.caption("Data: Cricsheet.org · IPL 2008–2026")

# ── DB guard ─────────────────────────────────────────────────────────────────
require_db(cached_db_ok())

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center; color:#1A3A5C;'>🏏 CricketIQ</h1>
    <p style='text-align:center; color:#95A5A6; font-size:1.1rem;'>
        AI-powered IPL analytics & tactical intelligence
    </p>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── Database metrics ─────────────────────────────────────────────────────────
summary = cached_db_summary()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("🏟️ Matches",    f"{summary.get('total_matches',    0):,}")
with c2:
    st.metric("⚾ Deliveries",  f"{summary.get('total_deliveries', 0):,}")
with c3:
    st.metric("👤 Players",    f"{summary.get('total_players',    0):,}")
with c4:
    st.metric("🏢 Teams",      f"{summary.get('total_teams',      0):,}")
with c5:
    earliest = str(summary.get("earliest_match", ""))[:4]
    latest   = str(summary.get("latest_match",   ""))[:4]
    st.metric("📅 Seasons",    f"{earliest} – {latest}")

st.divider()

# ── Quick navigation cards ────────────────────────────────────────────────────
st.subheader("🚀 Explore")
n1, n2, n3 = st.columns(3)

with n1:
    st.markdown(
        """
        <div style='background:#F0F4F8; border-radius:12px; padding:20px; text-align:center;
                    border-left:4px solid #1A3A5C; min-height:130px; display:flex;
                    flex-direction:column; justify-content:center;'>
            <h3 style='margin:0 0 8px 0;'>🏏 Player Intelligence</h3>
            <p style='color:#555; margin:0;'>Career stats, phase breakdowns, season trends, dismissal analysis for any IPL player.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with n2:
    st.markdown(
        """
        <div style='background:#F0F4F8; border-radius:12px; padding:20px; text-align:center;
                    border-left:4px solid #F4A300; min-height:130px; display:flex;
                    flex-direction:column; justify-content:center;'>
            <h3 style='margin:0 0 8px 0;'>⚔️ Matchup Engine</h3>
            <p style='color:#555; margin:0;'>Head-to-head analysis between any batter and bowler across all IPL seasons.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with n3:
    st.markdown(
        """
        <div style='background:#F0F4F8; border-radius:12px; padding:20px; text-align:center;
                    border-left:4px solid #2ECC71; min-height:130px; display:flex;
                    flex-direction:column; justify-content:center;'>
            <h3 style='margin:0 0 8px 0;'>🏆 Team Intelligence</h3>
            <p style='color:#555; margin:0;'>Win rates, head-to-head records, venue analysis, toss impact, phase scoring patterns.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── Recent matches ────────────────────────────────────────────────────────────
st.subheader("📋 Recent Matches")

seasons_df = cached_seasons()
season_options = ["All seasons"] + (seasons_df["season"].tolist() if not seasons_df.empty else [])
selected_season_label = st.selectbox("Filter by season", season_options, index=0)

season_id = None
if selected_season_label != "All seasons" and not seasons_df.empty:
    row = seasons_df[seasons_df["season"] == selected_season_label]
    if not row.empty:
        season_id = int(row["id"].iloc[0])

with st.spinner("Loading matches..."):
    matches_df = cached_matches(season_id=season_id)

if matches_df.empty:
    st.info("No matches found.")
else:
    # Build display table
    display = matches_df[["match_date", "team1_name", "team2_name", "winner_name",
                            "win_margin_runs", "win_margin_wickets", "dls_applied",
                            "venue_name", "season"]].copy()
    display["Result"] = display.apply(format_result, axis=1)
    display["Match"]  = display["team1_name"] + " vs " + display["team2_name"]
    display["Date"]   = pd.to_datetime(display["match_date"]).dt.strftime("%d %b %Y")

    show = display[["Date", "Match", "Result", "venue_name", "season"]].rename(
        columns={"venue_name": "Venue", "season": "Season"}
    )
    st.dataframe(
        show.head(20),
        use_container_width=True,
        hide_index=True,
    )
    if len(matches_df) > 20:
        st.caption(f"Showing 20 of {len(matches_df):,} matches. Use season filter to narrow down.")

st.divider()

# ── Fun stat ─────────────────────────────────────────────────────────────────
st.subheader("📈 Quick Stat")

players_df = cached_players()
if not players_df.empty and not matches_df.empty:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Most active seasons in dataset**")
        if "season" in matches_df.columns:
            season_counts = (
                matches_df["season"]
                .value_counts()
                .head(5)
                .reset_index()
                .rename(columns={"season": "Season", "count": "Matches"})
            )
            st.dataframe(season_counts, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Venues with most matches**")
        if "venue_name" in matches_df.columns:
            venue_counts = (
                matches_df["venue_name"]
                .value_counts()
                .head(5)
                .reset_index()
                .rename(columns={"venue_name": "Venue", "count": "Matches"})
            )
            st.dataframe(venue_counts, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "CricketIQ · Data source: Cricsheet.org · "
    "Built with Python, PostgreSQL, pandas, matplotlib, Streamlit"
)
