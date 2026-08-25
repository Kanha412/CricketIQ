"""
CricketIQ — Team Intelligence Page
=====================================
Team win rates, head-to-head records, toss impact, venue analysis.

Layout:
  Tabs:
    1. Overview     — win rates chart + season records table
    2. Head-to-Head — pick 2 teams → split bar + stats
    3. Venues       — venue win % chart + phase scoring
    4. Toss Impact  — toss analysis bars
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import app.cloud  # noqa: F401 — Streamlit secrets → env vars bridge (cloud deployment)

import streamlit as st
import pandas as pd

from app.utils import cached_db_ok, cached_matches, cached_seasons
from app.components import require_db, no_data_message, section

from src.analytics.team import (
    get_team_win_rates,
    get_head_to_head,
    get_toss_impact,
    get_venue_summary,
    get_team_season_record,
)
from src.visualization.team_charts import (
    win_rate_chart,
    head_to_head_chart,
    toss_impact_chart,
    venue_win_rate_chart,
)
from src.visualization.style import apply_theme

apply_theme()
require_db(cached_db_ok())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏆 Team Intelligence")
    st.divider()

    seasons_df    = cached_seasons()
    season_options = ["All seasons"] + (seasons_df["season"].tolist() if not seasons_df.empty else [])
    selected_season = st.selectbox("📅 Season", season_options)

    season_id = None
    if selected_season != "All seasons" and not seasons_df.empty:
        row = seasons_df[seasons_df["season"] == selected_season]
        if not row.empty:
            season_id = int(row["id"].iloc[0])

    st.divider()
    st.caption("Data: IPL 2008–2024 · Cricsheet.org")

# ── Load matches ──────────────────────────────────────────────────────────────
st.title("🏆 Team Intelligence")

with st.spinner("Loading match data..."):
    matches_df = cached_matches(season_id=season_id)

if matches_df.empty:
    no_data_message("the selected season")
    st.stop()

season_label = selected_season if selected_season != "All seasons" else "All Seasons"
st.caption(f"IPL · {season_label} · {len(matches_df):,} matches")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_h2h, tab_venue, tab_toss = st.tabs([
    "📊 Overview", "⚔️ Head-to-Head", "🏟️ Venues", "🎲 Toss Impact"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    section("Team Win Rates", "📊")
    win_df = get_team_win_rates(matches_df)

    if win_df.empty:
        no_data_message("win rate data")
    else:
        st.pyplot(win_rate_chart(win_df, f"IPL Win Rates — {season_label}"), use_container_width=True)

        # Top 3 teams
        top3 = win_df.head(3)
        st.markdown("**🥇 Top performers**")
        c1, c2, c3 = st.columns(3)
        for col, (_, row) in zip([c1, c2, c3], top3.iterrows()):
            col.metric(row["team"], f"{row['win_pct']:.1f}%", f"{int(row['won'])}W / {int(row['played'])}P")

    st.divider()

    # Season record for a selected team
    section("Season Record", "📅")
    teams_list = sorted(matches_df["team1_name"].dropna().unique().tolist())
    default_team = "Mumbai Indians" if "Mumbai Indians" in teams_list else teams_list[0] if teams_list else None

    if default_team:
        selected_team = st.selectbox("Select team for season breakdown", teams_list,
                                      index=teams_list.index(default_team) if default_team in teams_list else 0)
        season_record = get_team_season_record(matches_df, selected_team)
        if not season_record.empty:
            st.dataframe(
                season_record.rename(columns={"season": "Season", "played": "P", "won": "W",
                                               "lost": "L", "win_pct": "Win%"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(f"No season data for {selected_team}.")

    st.divider()
    st.dataframe(
        win_df.rename(columns={"team": "Team", "played": "Played",
                                "won": "Won", "win_pct": "Win%"}),
        use_container_width=True,
        hide_index=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HEAD-TO-HEAD
# ══════════════════════════════════════════════════════════════════════════════
with tab_h2h:
    section("Head-to-Head", "⚔️")

    teams_list = sorted(matches_df["team1_name"].dropna().unique().tolist())
    col_a, col_b = st.columns(2)
    with col_a:
        default_a = teams_list.index("Mumbai Indians") if "Mumbai Indians" in teams_list else 0
        team_a = st.selectbox("🔵 Team A", teams_list, index=default_a, key="h2h_a")
    with col_b:
        default_b = teams_list.index("Chennai Super Kings") if "Chennai Super Kings" in teams_list else 1
        team_b = st.selectbox("🟡 Team B", teams_list, index=default_b, key="h2h_b")

    if team_a == team_b:
        st.warning("Please select two different teams.")
    else:
        h2h = get_head_to_head(matches_df, team_a, team_b)
        st.pyplot(head_to_head_chart(h2h, team_a, team_b), use_container_width=True)

        total = h2h.get("total_matches", 0)
        if total > 0:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Matches",       total)
            m2.metric(f"{team_a} wins",      h2h.get("team_a_wins", 0),
                       f"{h2h.get('team_a_win_pct', 0):.1f}%")
            m3.metric(f"{team_b} wins",      h2h.get("team_b_wins", 0),
                       f"{h2h.get('team_b_win_pct', 0):.1f}%")
            m4.metric("No Result",           h2h.get("no_result", 0))

            # Recent H2H matches table
            st.divider()
            section("Recent Meetings", "📋")
            h2h_matches = matches_df[
                (
                    (matches_df["team1_name"] == team_a) & (matches_df["team2_name"] == team_b)
                ) | (
                    (matches_df["team1_name"] == team_b) & (matches_df["team2_name"] == team_a)
                )
            ].copy()
            if "match_date" in h2h_matches.columns:
                h2h_matches = h2h_matches.sort_values("match_date", ascending=False)
            show_cols = [c for c in ["match_date", "team1_name", "team2_name", "winner_name",
                                      "venue_name", "season"] if c in h2h_matches.columns]
            st.dataframe(
                h2h_matches[show_cols].rename(columns={
                    "match_date": "Date", "team1_name": "Team 1", "team2_name": "Team 2",
                    "winner_name": "Winner", "venue_name": "Venue", "season": "Season"
                }).head(10),
                use_container_width=True,
                hide_index=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VENUES
# ══════════════════════════════════════════════════════════════════════════════
with tab_venue:
    section("Venue Analysis", "🏟️")

    venue_df = get_venue_summary(matches_df)

    if venue_df.empty:
        no_data_message("venue data")
    else:
        top_n = st.slider("Show top N venues", 5, min(20, len(venue_df)), 10)
        st.pyplot(venue_win_rate_chart(venue_df, top_n=top_n), use_container_width=True)
        st.caption(
            "🔵 Batting first win % vs 🟡 Bowling first win % at each venue. "
            "Closer to 50% = more balanced pitch."
        )

        st.divider()
        st.markdown("**Full venue table**")
        show_cols = [c for c in ["venue_name", "matches", "bat_first_wins",
                                   "bowl_first_wins", "bat_first_win_pct"] if c in venue_df.columns]
        st.dataframe(
            venue_df[show_cols].rename(columns={
                "venue_name": "Venue", "matches": "Matches",
                "bat_first_wins": "Bat First Wins", "bowl_first_wins": "Bowl First Wins",
                "bat_first_win_pct": "Bat First Win%"
            }),
            use_container_width=True,
            hide_index=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TOSS IMPACT
# ══════════════════════════════════════════════════════════════════════════════
with tab_toss:
    section("Toss Impact Analysis", "🎲")

    # Need toss_winner and toss_decision — check if columns exist
    # They're not in get_matches() — need to add them or use a fallback message
    if "toss_winner" not in matches_df.columns or "toss_decision" not in matches_df.columns:
        st.info(
            "⚠️ Toss data not available in the current matches query. "
            "The `toss_winner` and `toss_decision` columns are stored in the database "
            "but not fetched by the current query. This will be added in V2."
        )
        st.markdown(
            """
            **What toss analysis will show:**
            - % of times toss winner also wins the match
            - Win rate when choosing to bat first vs field first
            - Venue-specific toss preferences
            """
        )
    else:
        toss_data = get_toss_impact(matches_df)
        if not toss_data:
            no_data_message("toss data")
        else:
            st.pyplot(toss_impact_chart(toss_data, f"Toss Impact — {season_label}"), use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Toss winner wins match",
                       f"{toss_data.get('toss_winner_won_match_pct', 0):.1f}%")
            m2.metric("Bat first win %",
                       f"{toss_data.get('bat_first_win_pct', 0):.1f}%" if toss_data.get('bat_first_win_pct') else "—")
            m3.metric("Bowl first win %",
                       f"{toss_data.get('field_first_win_pct', 0):.1f}%" if toss_data.get('field_first_win_pct') else "—")
            st.caption(
                f"Based on {toss_data.get('total_matches_analysed', 0):,} matches. "
                "50% = no advantage — toss is a coin flip in terms of winning."
            )

st.divider()
st.caption("CricketIQ · Data: Cricsheet.org · IPL 2008–2024")
