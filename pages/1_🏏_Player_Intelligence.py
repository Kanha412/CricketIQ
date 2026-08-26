"""
CricketIQ — Player Intelligence Page
======================================
Full batting and bowling analytics for any IPL player.

Layout:
  Sidebar  — player search, role toggle, season filter
  Main     — headline stats → phase chart + dismissal/wicket donut
           → season trend → over-by-over map → vs-IPL-avg bars
           → raw data expander
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import app.cloud  # noqa: F401 — Streamlit secrets → env vars bridge (cloud deployment)

import streamlit as st

from app.utils import (
    cached_db_ok,
    cached_player_deliveries,
    cached_players,
    cached_seasons,
    build_name_id_map,
)
from app.components import require_db, no_data_message, section

from src.analytics.batting import (
    get_batting_summary,
    get_phase_batting_stats,
    get_batting_season_trend,
    get_scoring_by_over,
    get_dismissal_breakdown,
    get_batting_by_venue,
)
from src.analytics.bowling import (
    get_bowling_summary,
    get_phase_bowling_stats,
    get_bowling_season_trend,
    get_economy_by_over,
    get_wicket_type_breakdown,
)
from src.visualization.batting_charts import (
    phase_batting_chart,
    batting_season_trend_chart,
    scoring_by_over_chart,
    dismissal_donut_chart,
    batting_summary_bars,
)
from src.visualization.bowling_charts import (
    phase_bowling_chart,
    bowling_season_trend_chart,
    economy_by_over_chart,
    wicket_type_chart,
    bowling_summary_bars,
)
from src.visualization.style import apply_theme

apply_theme()
require_db(cached_db_ok())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏏 Player Intelligence")
    st.divider()

    # Player search
    players_df = cached_players()
    if players_df.empty:
        st.error("No players in database.")
        st.stop()

    name_to_id = build_name_id_map(players_df)
    player_names = players_df["name"].tolist()

    # Default to V Kohli if available
    default_idx = player_names.index("V Kohli") if "V Kohli" in player_names else 0
    selected_name = st.selectbox("🔍 Select player", player_names, index=default_idx)
    player_id = name_to_id[selected_name]

    # Role
    role = st.radio("Role", ["batter", "bowler"], horizontal=True)

    # Season filter
    st.divider()
    seasons_df = cached_seasons()
    season_options = ["All seasons"] + (seasons_df["season"].tolist() if not seasons_df.empty else [])
    selected_season = st.selectbox("📅 Season", season_options)

    season_id = None
    if selected_season != "All seasons" and not seasons_df.empty:
        row = seasons_df[seasons_df["season"] == selected_season]
        if not row.empty:
            season_id = int(row["id"].iloc[0])

    st.divider()
    st.caption("Data: IPL 2008–2026 · Cricsheet.org")

# ── Load data ─────────────────────────────────────────────────────────────────
st.title(f"🏏 {selected_name}")
season_label = selected_season if selected_season != "All seasons" else "All Seasons"
st.caption(f"IPL · {season_label} · {role.title()}")

with st.spinner(f"Loading {selected_name}'s deliveries..."):
    df = cached_player_deliveries(player_id, role, season_id=season_id)

if df.empty:
    no_data_message(f"{selected_name} as {role} ({season_label})")
    st.stop()

st.success(f"**{len(df):,}** deliveries · **{df['match_id'].nunique()}** matches")
st.divider()

# ── BATTER view ───────────────────────────────────────────────────────────────
if role == "batter":
    summary    = get_batting_summary(df)
    phase_df   = get_phase_batting_stats(df)
    season_df  = get_batting_season_trend(df)
    over_df    = get_scoring_by_over(df)
    dismiss_df = get_dismissal_breakdown(df)

    # Headline metrics
    section("Career Summary", "📊")
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Runs",        f"{summary.get('runs') or 0:,}")
    m2.metric("Innings",     summary.get("innings") or "—")
    m3.metric("Average",     f"{summary.get('avg') or 0:.1f}" if summary.get("avg") else "∞")
    m4.metric("Strike Rate", f"{summary.get('sr') or 0:.1f}"  if summary.get("sr")  else "—")
    m5.metric("4s / 6s",     f"{summary.get('fours') or 0} / {summary.get('sixes') or 0}")
    m6.metric("50s / 100s",  f"{summary.get('fifties') or 0} / {summary.get('hundreds') or 0}")
    m7.metric("Best",        summary.get("highest_score") or "—")

    st.divider()

    # Phase + Dismissals side by side
    section("Phase & Dismissal Breakdown", "📐")
    ch1, ch2 = st.columns([3, 2])
    with ch1:
        st.pyplot(phase_batting_chart(phase_df, selected_name), use_container_width=True)
    with ch2:
        st.pyplot(dismissal_donut_chart(dismiss_df, selected_name), use_container_width=True)

    # Phase data table
    with st.expander("📋 Phase stats (raw)"):
        if not phase_df.empty:
            display_cols = [c for c in ["phase_label", "balls_faced", "runs", "dismissals",
                                         "avg", "sr", "dot_pct", "boundary_pct"] if c in phase_df.columns]
            st.dataframe(phase_df[display_cols], use_container_width=True, hide_index=True)

    st.divider()

    # Season trend
    section("Season Trend", "📈")
    st.pyplot(batting_season_trend_chart(season_df, selected_name), use_container_width=True)

    st.divider()

    # Scoring map
    section("Scoring Map by Over", "🗺️")
    st.pyplot(scoring_by_over_chart(over_df, selected_name), use_container_width=True)

    st.divider()

    # vs IPL Avg
    section("vs IPL Average", "⚖️")
    st.pyplot(batting_summary_bars(summary, selected_name), use_container_width=True)
    st.caption(
        "IPL benchmarks: SR ≈ 135 · Average ≈ 28 · Boundary% ≈ 17% · "
        "Dot-ball% inverted (higher = fewer dots = better)"
    )

    # Venue breakdown
    st.divider()
    section("Venue Performance", "🏟️")
    venue_df = get_batting_by_venue(df)
    if not venue_df.empty:
        display_cols = [c for c in ["venue_name", "matches", "runs", "balls_faced", "avg", "sr"] if c in venue_df.columns]
        st.dataframe(venue_df[display_cols].head(10), use_container_width=True, hide_index=True)
    else:
        st.info("Venue data not available for this selection.")

# ── BOWLER view ───────────────────────────────────────────────────────────────
else:
    summary   = get_bowling_summary(df)
    phase_df  = get_phase_bowling_stats(df)
    season_df = get_bowling_season_trend(df)
    over_df   = get_economy_by_over(df)
    wkt_df    = get_wicket_type_breakdown(df)

    # Headline metrics
    section("Career Summary", "📊")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Wickets",    f"{summary.get('wickets') or 0:,}")
    m2.metric("Matches",    summary.get("matches") or "—")
    m3.metric("Economy",    f"{summary.get('economy') or 0:.2f}" if summary.get("economy") else "—")
    m4.metric("Average",    f"{summary.get('avg') or 0:.1f}"     if summary.get("avg")      else "—")
    m5.metric("SR",         f"{summary.get('sr') or 0:.1f}"      if summary.get("sr")       else "—")
    m6.metric("Best",       summary.get("best_bowling") or "—")

    st.divider()

    # Phase + Wicket type
    section("Phase & Wicket Breakdown", "📐")
    ch1, ch2 = st.columns([3, 2])
    with ch1:
        st.pyplot(phase_bowling_chart(phase_df, selected_name), use_container_width=True)
    with ch2:
        st.pyplot(wicket_type_chart(wkt_df, selected_name), use_container_width=True)

    with st.expander("📋 Phase stats (raw)"):
        if not phase_df.empty:
            display_cols = [c for c in ["phase_label", "overs", "runs_conceded", "wickets",
                                         "economy", "avg", "dot_pct"] if c in phase_df.columns]
            st.dataframe(phase_df[display_cols], use_container_width=True, hide_index=True)

    st.divider()

    # Season trend
    section("Season Trend", "📈")
    st.pyplot(bowling_season_trend_chart(season_df, selected_name), use_container_width=True)

    st.divider()

    # Economy by over
    section("Economy by Over", "🎯")
    st.pyplot(economy_by_over_chart(over_df, selected_name), use_container_width=True)
    st.caption("🟢 Below IPL avg (8.0) = good  ·  🔴 Above = costly")

    st.divider()

    # vs IPL Avg
    section("vs IPL Average", "⚖️")
    st.pyplot(bowling_summary_bars(summary, selected_name), use_container_width=True)
    st.caption("IPL benchmarks: Economy ≈ 8.0 · Average ≈ 30 · SR ≈ 18 · Dot% ≈ 38%")

st.divider()
st.caption("CricketIQ · Data: Cricsheet.org · IPL 2008–2026")
