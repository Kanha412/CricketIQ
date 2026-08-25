"""
CricketIQ — Database Query Functions
======================================
All database reads go through this file. No raw SQL in analytics modules.

DESIGN PRINCIPLES:
    1. Every function takes IDs, returns Pandas DataFrames or typed dicts
    2. All queries use SQLAlchemy text() with named parameters (no string formatting)
    3. Functions are decorated with @st.cache_data in Streamlit pages (not here)
    4. Heavy queries have a `limit` parameter to prevent memory overload
    5. Returns empty DataFrame (not None) on no results — callers don't need null checks

NAMING CONVENTION:
    get_*     → returns DataFrame or dict (read-only)
    list_*    → returns list of summary objects
    exists_*  → returns bool
"""

import pandas as pd
from sqlalchemy import text

from src.db.connection import get_connection
from src.logger import get_logger

logger = get_logger(__name__)


# ── Competition / Season / Match Lookups ──────────────────────────────────────

def get_competitions() -> pd.DataFrame:
    """
    Return all competitions in the database.

    Returns:
        DataFrame with columns: id, name, short_name, gender, match_type
    """
    with get_connection() as conn:
        return pd.read_sql(
            text("SELECT id, name, short_name, gender, match_type FROM competitions ORDER BY name"),
            conn,
        )


def get_seasons(competition_id: int) -> pd.DataFrame:
    """
    Return all seasons for a competition, most recent first.

    Returns:
        DataFrame with columns: id, season
    """
    with get_connection() as conn:
        return pd.read_sql(
            text("""
                SELECT id, season
                FROM seasons
                WHERE competition_id = :comp_id
                ORDER BY season DESC
            """),
            conn,
            params={"comp_id": competition_id},
        )


def get_matches(
    competition_id: int | None = None,
    season_id: int | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """
    Return matches, optionally filtered by competition and season.

    Returns:
        DataFrame with match summaries (id, date, teams, result, winner, venue)
    """
    with get_connection() as conn:
        return pd.read_sql(
            text("""
                SELECT
                    m.id,
                    m.cricsheet_file_id,
                    m.match_date,
                    m.match_type,
                    m.stage,
                    m.match_number,
                    t1.name  AS team1,
                    t2.name  AS team2,
                    tw.name  AS winner,
                    m.win_margin_runs,
                    m.win_margin_wickets,
                    m.result,
                    m.dls_applied,
                    v.name   AS venue,
                    v.city,
                    s.season
                FROM matches m
                LEFT JOIN teams  t1 ON t1.id = m.team1_id
                LEFT JOIN teams  t2 ON t2.id = m.team2_id
                LEFT JOIN teams  tw ON tw.id = m.winner_id
                LEFT JOIN venues  v ON v.id  = m.venue_id
                LEFT JOIN seasons s ON s.id  = m.season_id
                WHERE
                    (:comp_id  IS NULL OR m.competition_id = :comp_id)
                    AND (:season_id IS NULL OR m.season_id  = :season_id)
                ORDER BY m.match_date DESC
                LIMIT :lim
            """),
            conn,
            params={
                "comp_id": competition_id,
                "season_id": season_id,
                "lim": limit,
            },
        )


def get_match_by_id(match_id: int) -> dict | None:
    """
    Return a single match's details as a dict.
    Returns None if match not found.
    """
    with get_connection() as conn:
        result = conn.execute(
            text("""
                SELECT
                    m.id, m.cricsheet_file_id, m.match_date, m.match_type,
                    m.stage, m.match_number, m.result, m.dls_applied,
                    m.win_margin_runs, m.win_margin_wickets,
                    t1.name  AS team1,     t1.id AS team1_id,
                    t2.name  AS team2,     t2.id AS team2_id,
                    tw.name  AS winner,    tw.id AS winner_id,
                    tt.name  AS toss_winner,
                    m.toss_decision,
                    v.name   AS venue,     v.city,
                    s.season,
                    p.name   AS player_of_match
                FROM matches m
                LEFT JOIN teams   t1 ON t1.id = m.team1_id
                LEFT JOIN teams   t2 ON t2.id = m.team2_id
                LEFT JOIN teams   tw ON tw.id = m.winner_id
                LEFT JOIN teams   tt ON tt.id = m.toss_winner_id
                LEFT JOIN venues   v ON v.id  = m.venue_id
                LEFT JOIN seasons  s ON s.id  = m.season_id
                LEFT JOIN players  p ON p.id  = m.player_of_match_id
                WHERE m.id = :match_id
            """),
            {"match_id": match_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None


def get_innings_for_match(match_id: int) -> pd.DataFrame:
    """
    Return innings summary for a match.

    Returns:
        DataFrame with columns: id, innings_number, batting_team, bowling_team,
                                 total_runs, total_wickets, total_overs, target_runs
    """
    with get_connection() as conn:
        return pd.read_sql(
            text("""
                SELECT
                    i.id,
                    i.innings_number,
                    tb.name AS batting_team,
                    tw.name AS bowling_team,
                    i.total_runs,
                    i.total_wickets,
                    i.total_overs,
                    i.target_runs,
                    i.is_super_over
                FROM innings i
                LEFT JOIN teams tb ON tb.id = i.batting_team_id
                LEFT JOIN teams tw ON tw.id = i.bowling_team_id
                WHERE i.match_id = :match_id
                ORDER BY i.innings_number
            """),
            conn,
            params={"match_id": match_id},
        )


# ── Delivery Queries (the analytical core) ────────────────────────────────────

def get_match_deliveries(match_id: int) -> pd.DataFrame:
    """
    Return ALL deliveries for a match with player names resolved.
    This is the primary input for match-level analytics.

    Cached in Streamlit pages with @st.cache_data(ttl=3600).

    Returns:
        DataFrame with ~250 rows (one per delivery), all derived fields included.
    """
    logger.debug("Loading deliveries for match %d", match_id)
    with get_connection() as conn:
        return pd.read_sql(
            text("""
                SELECT
                    d.id,
                    d.innings_id,
                    d.match_id,
                    d.over_number,
                    d.ball_in_over,
                    d.legal_ball_number,
                    i.innings_number,

                    -- Player names (joined for readability in analytics)
                    pb.name   AS batter,
                    pbo.name  AS bowler,
                    pns.name  AS non_striker,

                    -- Player IDs (for cross-match lookups)
                    d.batter_id,
                    d.bowler_id,
                    d.non_striker_id,

                    -- Runs
                    d.runs_batter,
                    d.runs_extras,
                    d.runs_total,
                    d.extras_type,

                    -- Wickets
                    d.is_wicket,
                    d.wicket_kind,
                    po.name   AS player_out,
                    d.player_out_id,

                    -- Derived fields
                    d.phase,
                    d.is_boundary,
                    d.is_six,
                    d.is_dot_ball,
                    d.is_legal_delivery,
                    d.cumulative_runs,
                    d.cumulative_wickets,

                    -- Team context (from innings)
                    tb.name   AS batting_team,
                    tw.name   AS bowling_team,
                    i.target_runs
                FROM deliveries d
                JOIN innings i   ON i.id  = d.innings_id
                JOIN players pb  ON pb.id = d.batter_id
                JOIN players pbo ON pbo.id = d.bowler_id
                LEFT JOIN players pns ON pns.id = d.non_striker_id
                LEFT JOIN players po  ON po.id  = d.player_out_id
                LEFT JOIN teams tb    ON tb.id  = i.batting_team_id
                LEFT JOIN teams tw    ON tw.id  = i.bowling_team_id
                WHERE d.match_id = :match_id
                ORDER BY d.innings_id, d.over_number, d.ball_in_over
            """),
            conn,
            params={"match_id": match_id},
        )


def get_player_deliveries(
    player_id: int,
    role: str = "batter",           # 'batter' or 'bowler'
    competition_id: int | None = None,
    season_id: int | None = None,
) -> pd.DataFrame:
    """
    Return all deliveries involving a specific player (batting or bowling).
    Used for player intelligence and rolling form calculations.

    Args:
        player_id:       Database ID of the player.
        role:            'batter' to get batting deliveries, 'bowler' for bowling.
        competition_id:  Filter to specific competition.
        season_id:       Filter to specific season.

    Returns:
        DataFrame of deliveries with match date for temporal sorting.
    """
    id_col = "d.batter_id" if role == "batter" else "d.bowler_id"
    with get_connection() as conn:
        return pd.read_sql(
            text(f"""
                SELECT
                    d.id,
                    d.match_id,
                    m.match_date,
                    s.season,
                    d.over_number,
                    d.phase,
                    d.runs_batter,
                    d.runs_total,
                    d.is_wicket,
                    d.wicket_kind,
                    d.player_out_id,
                    d.batter_id,
                    d.bowler_id,
                    d.is_boundary,
                    d.is_six,
                    d.is_dot_ball,
                    d.is_legal_delivery,
                    d.extras_type,
                    d.cumulative_runs
                FROM deliveries d
                JOIN matches m ON m.id = d.match_id
                LEFT JOIN seasons s ON s.id = m.season_id
                WHERE {id_col} = :player_id
                    AND (:comp_id IS NULL OR m.competition_id = :comp_id)
                    AND (:season_id IS NULL OR m.season_id = :season_id)
                ORDER BY m.match_date, d.innings_id, d.over_number, d.ball_in_over
            """),
            conn,
            params={
                "player_id": player_id,
                "comp_id": competition_id,
                "season_id": season_id,
            },
        )


def get_batter_bowler_deliveries(
    batter_id: int,
    bowler_id: int,
    competition_id: int | None = None,
) -> pd.DataFrame:
    """
    Return all deliveries between a specific batter and bowler.
    Core function for the Matchup Engine.

    Returns:
        DataFrame of head-to-head deliveries with context.
    """
    with get_connection() as conn:
        return pd.read_sql(
            text("""
                SELECT
                    d.id,
                    d.match_id,
                    m.match_date,
                    s.season,
                    d.over_number,
                    d.phase,
                    d.runs_batter,
                    d.runs_total,
                    d.is_wicket,
                    d.wicket_kind,
                    d.is_boundary,
                    d.is_six,
                    d.is_dot_ball,
                    d.is_legal_delivery,
                    d.extras_type
                FROM deliveries d
                JOIN matches m ON m.id = d.match_id
                LEFT JOIN seasons s ON s.id = m.season_id
                WHERE d.batter_id  = :batter_id
                  AND d.bowler_id  = :bowler_id
                  AND d.is_legal_delivery = true
                  AND (:comp_id IS NULL OR m.competition_id = :comp_id)
                ORDER BY m.match_date, d.over_number
            """),
            conn,
            params={
                "batter_id": batter_id,
                "bowler_id": bowler_id,
                "comp_id": competition_id,
            },
        )


# ── Player Lookups ────────────────────────────────────────────────────────────

def get_all_players(competition_id: int | None = None) -> pd.DataFrame:
    """
    Return all players, optionally filtered to those who have played
    in a specific competition.

    Returns:
        DataFrame with columns: id, name, cricsheet_id
    """
    if competition_id is None:
        with get_connection() as conn:
            return pd.read_sql(
                text("SELECT id, name, cricsheet_id FROM players ORDER BY name"),
                conn,
            )
    else:
        with get_connection() as conn:
            return pd.read_sql(
                text("""
                    SELECT DISTINCT p.id, p.name, p.cricsheet_id
                    FROM players p
                    JOIN deliveries d ON d.batter_id = p.id
                    JOIN matches    m ON m.id = d.match_id
                    WHERE m.competition_id = :comp_id
                    ORDER BY p.name
                """),
                conn,
                params={"comp_id": competition_id},
            )


def get_player_by_id(player_id: int) -> dict | None:
    """Return a player's details as a dict. None if not found."""
    with get_connection() as conn:
        result = conn.execute(
            text("SELECT id, name, cricsheet_id, batting_style, bowling_style, role FROM players WHERE id = :pid"),
            {"pid": player_id},
        )
        row = result.mappings().first()
        return dict(row) if row else None


def search_players(query: str, limit: int = 20) -> pd.DataFrame:
    """
    Search players by name (case-insensitive partial match).
    Used for the player selector in Streamlit UI.
    """
    with get_connection() as conn:
        return pd.read_sql(
            text("""
                SELECT id, name, cricsheet_id
                FROM players
                WHERE LOWER(name) LIKE LOWER(:q)
                ORDER BY name
                LIMIT :lim
            """),
            conn,
            params={"q": f"%{query}%", "lim": limit},
        )


# ── Aggregated Stats ──────────────────────────────────────────────────────────

def get_player_match_stats(
    player_id: int,
    competition_id: int | None = None,
    season_id: int | None = None,
) -> pd.DataFrame:
    """
    Return per-match stats for a player from the pre-computed table.
    Much faster than aggregating from deliveries for career stats.
    """
    with get_connection() as conn:
        return pd.read_sql(
            text("""
                SELECT
                    pms.*,
                    m.match_date,
                    s.season,
                    t1.name AS team1,
                    t2.name AS team2,
                    pt.name AS team
                FROM player_match_stats pms
                JOIN matches m ON m.id = pms.match_id
                LEFT JOIN seasons s ON s.id = m.season_id
                LEFT JOIN teams t1 ON t1.id = m.team1_id
                LEFT JOIN teams t2 ON t2.id = m.team2_id
                LEFT JOIN teams pt ON pt.id = pms.team_id
                WHERE pms.player_id = :player_id
                    AND (:comp_id IS NULL OR m.competition_id = :comp_id)
                    AND (:season_id IS NULL OR m.season_id = :season_id)
                ORDER BY m.match_date
            """),
            conn,
            params={
                "player_id": player_id,
                "comp_id": competition_id,
                "season_id": season_id,
            },
        )


# ── Database Statistics ───────────────────────────────────────────────────────

def get_database_summary() -> dict:
    """
    Return high-level counts for the Home page dashboard.
    """
    with get_connection() as conn:
        result = conn.execute(text("""
            SELECT
                (SELECT COUNT(*) FROM matches)       AS total_matches,
                (SELECT COUNT(*) FROM players)       AS total_players,
                (SELECT COUNT(*) FROM teams)         AS total_teams,
                (SELECT COUNT(*) FROM deliveries)    AS total_deliveries,
                (SELECT COUNT(*) FROM competitions)  AS total_competitions,
                (SELECT MIN(match_date) FROM matches) AS earliest_match,
                (SELECT MAX(match_date) FROM matches) AS latest_match
        """))
        row = result.mappings().first()
        return dict(row) if row else {}


def get_ingested_file_ids() -> set[str]:
    """
    Return set of cricsheet_file_ids already in the database.
    Used by the ingestion script to skip already-processed files.
    """
    with get_connection() as conn:
        result = conn.execute(text("SELECT cricsheet_file_id FROM matches"))
        return {row[0] for row in result}
