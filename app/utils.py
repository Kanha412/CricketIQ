"""
CricketIQ — Shared Streamlit Cache Layer
==========================================
All database calls in the app go through here.
Every function is decorated with @st.cache_data so the DB is hit only
once per TTL window even if multiple pages / reruns call the same function.

WHY A SEPARATE utils.py:
    Without this, every page file would duplicate the same @st.cache_data
    wrappers.  Centralising here means one TTL change propagates everywhere,
    and we have one place to see what data the UI depends on.

TTL strategy:
    - Player lists, competitions: 10 min (rarely change mid-session)
    - Player deliveries:          5 min  (slightly stale is fine)
    - Match data:                 10 min
    - DB summary:                 2 min  (home-page headline, should be fresh)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import test_connection
from src.db.queries import (
    get_all_players,
    get_batter_bowler_deliveries,
    get_competitions,
    get_database_summary,
    get_matches,
    get_player_deliveries,
    get_seasons,
)


# ---------------------------------------------------------------------------
# Connection health
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def cached_db_ok() -> bool:
    return test_connection()


# ---------------------------------------------------------------------------
# Reference data (slow-changing)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600)
def cached_db_summary() -> dict:
    return get_database_summary()


@st.cache_data(ttl=600)
def cached_competitions() -> pd.DataFrame:
    return get_competitions()


@st.cache_data(ttl=600)
def cached_seasons(competition_id: int | None = None) -> pd.DataFrame:
    return get_seasons(competition_id)


@st.cache_data(ttl=600)
def cached_players(competition_id: int | None = None) -> pd.DataFrame:
    """All players, sorted by name. Returns DataFrame with id, name columns."""
    df = get_all_players(competition_id=competition_id)
    if df.empty:
        return df
    return df.sort_values("name").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Match data (renamed for analytics layer)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600)
def cached_matches(competition_id: int | None = None, season_id: int | None = None) -> pd.DataFrame:
    """
    Returns all matches (up to 2000) with column names expected by analytics.team:
        team1_name, team2_name, winner_name, venue_name, toss_winner, toss_decision

    WHY RENAME HERE: get_matches() uses short aliases (team1, venue) for query
    readability; analytics.team uses long names for clarity.  The rename lives
    here so neither layer has to change.
    """
    df = get_matches(competition_id=competition_id, season_id=season_id, limit=2000)
    if df.empty:
        return df
    return df.rename(columns={
        "team1":  "team1_name",
        "team2":  "team2_name",
        "winner": "winner_name",
        "venue":  "venue_name",
    })


# ---------------------------------------------------------------------------
# Player deliveries (most-called query)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def cached_player_deliveries(
    player_id: int,
    role: str,
    competition_id: int | None = None,
    season_id: int | None = None,
) -> pd.DataFrame:
    return get_player_deliveries(
        player_id=player_id,
        role=role,
        competition_id=competition_id,
        season_id=season_id,
    )


# ---------------------------------------------------------------------------
# Matchup deliveries
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def cached_matchup_deliveries(
    batter_id: int,
    bowler_id: int,
    competition_id: int | None = None,
) -> pd.DataFrame:
    return get_batter_bowler_deliveries(
        batter_id=batter_id,
        bowler_id=bowler_id,
        competition_id=competition_id,
    )


# ---------------------------------------------------------------------------
# Helper: build name→id map from players DataFrame
# ---------------------------------------------------------------------------
def build_name_id_map(players_df: pd.DataFrame) -> dict[str, int]:
    """Returns {player_name: player_id} dict."""
    return dict(zip(players_df["name"], players_df["id"]))
