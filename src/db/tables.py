"""
CricketIQ — SQLAlchemy Core Table Definitions
===============================================
This file is the SINGLE SOURCE OF TRUTH for the database schema.

Both Alembic migrations and query code import from here.
Never define schema anywhere else.

DESIGN DECISIONS:
    - SQLAlchemy Core (not ORM): we need complex analytical SQL
    - JSONB for raw_json: allows re-parsing without schema changes
    - Derived columns in deliveries: pre-computed at ingestion for query speed
    - Materialized view defined here as a string: refreshed by a script
    - Separate player_match_stats table: pre-aggregated for dashboard speed

TABLE DEPENDENCY ORDER (for insertion):
    1. competitions  (no deps)
    2. venues        (no deps)
    3. teams         (no deps)
    4. players       (no deps)
    5. seasons       (→ competitions)
    6. matches       (→ competitions, seasons, venues, teams, players)
    7. innings       (→ matches, teams)
    8. deliveries    (→ innings, matches, players)
    9. dismissals    (→ deliveries, players)
    10. player_match_stats (→ matches, players, teams)
    11. analysis_runs (→ matches, players)
"""

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP

# Shared MetaData object — Alembic reads this to detect schema changes
metadata = MetaData()

# ── Reference Tables ──────────────────────────────────────────────────────────

competitions = Table(
    "competitions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(150), nullable=False, unique=True),
    Column("short_name", String(20)),
    Column("gender", String(10)),          # 'male', 'female'
    Column("match_type", String(20)),      # 'T20', 'ODI', 'Test'
)

venues = Table(
    "venues",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(250), nullable=False),
    Column("city", String(100)),
    Column("country", String(100)),
    UniqueConstraint("name", name="uq_venues_name"),
)

teams = Table(
    "teams",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(150), nullable=False, unique=True),
    Column("short_name", String(20)),
)

players = Table(
    "players",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("cricsheet_id", String(50), unique=True),   # Cricsheet registry ID
    Column("name", String(150), nullable=False),
    Column("batting_style", String(60)),                # Added later via update
    Column("bowling_style", String(60)),
    Column("role", String(50)),
    Column("nationality", String(100)),
)

# ── Match Tables ──────────────────────────────────────────────────────────────

seasons = Table(
    "seasons",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("competition_id", Integer, ForeignKey("competitions.id"), nullable=False),
    Column("season", String(20), nullable=False),       # '2023', '2022/23'
    UniqueConstraint("competition_id", "season", name="uq_seasons_comp_season"),
)

matches = Table(
    "matches",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("cricsheet_file_id", String(50), nullable=False, unique=True),  # filename no .json
    Column("competition_id", Integer, ForeignKey("competitions.id")),
    Column("season_id", Integer, ForeignKey("seasons.id")),
    Column("venue_id", Integer, ForeignKey("venues.id")),

    # Date
    Column("match_date", Date, nullable=False),

    # Match type
    Column("match_type", String(20), nullable=False),
    Column("gender", String(10)),

    # Teams
    Column("team1_id", Integer, ForeignKey("teams.id")),
    Column("team2_id", Integer, ForeignKey("teams.id")),

    # Toss
    Column("toss_winner_id", Integer, ForeignKey("teams.id")),
    Column("toss_decision", String(10)),               # 'bat' or 'field'

    # Result
    Column("winner_id", Integer, ForeignKey("teams.id")),
    Column("win_margin_runs", Integer),
    Column("win_margin_wickets", Integer),
    Column("result", String(20)),                      # 'win','tie','no_result','draw'
    Column("dls_applied", Boolean, server_default=text("false")),

    # Event info
    Column("event_name", String(150)),
    Column("match_number", Integer),
    Column("stage", String(60)),                       # 'Final', 'Qualifier 1', etc.

    # First player of match (simplification — stores first if multiple)
    Column("player_of_match_id", Integer, ForeignKey("players.id")),

    # Raw JSON for re-parsing (forward compatibility)
    Column("raw_json", JSONB),

    Column("ingested_at", TIMESTAMP, server_default=text("NOW()")),
)

innings = Table(
    "innings",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("match_id", Integer, ForeignKey("matches.id", ondelete="CASCADE"), nullable=False),
    Column("innings_number", SmallInteger, nullable=False),      # 1 or 2 (3/4 super overs)
    Column("batting_team_id", Integer, ForeignKey("teams.id")),
    Column("bowling_team_id", Integer, ForeignKey("teams.id")),
    Column("target_runs", Integer),                              # NULL for first innings
    Column("total_runs", Integer),
    Column("total_wickets", SmallInteger),
    Column("total_overs", Numeric(4, 1)),                        # e.g. 19.4
    Column("is_super_over", Boolean, server_default=text("false")),
    UniqueConstraint("match_id", "innings_number", name="uq_innings_match_innings"),
)

# ── Core Analytical Table ─────────────────────────────────────────────────────

deliveries = Table(
    "deliveries",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),

    # Position
    Column("innings_id", Integer, ForeignKey("innings.id", ondelete="CASCADE"), nullable=False),
    Column("match_id", Integer, ForeignKey("matches.id"), nullable=False),  # denormalised for speed
    Column("over_number", SmallInteger, nullable=False),          # 1-indexed
    Column("ball_in_over", SmallInteger, nullable=False),         # Sequence incl. wides/no-balls
    Column("legal_ball_number", SmallInteger),                    # 1-6 (advances per legal ball)

    # Players
    Column("batter_id", Integer, ForeignKey("players.id"), nullable=False),
    Column("non_striker_id", Integer, ForeignKey("players.id")),
    Column("bowler_id", Integer, ForeignKey("players.id"), nullable=False),

    # Runs
    Column("runs_batter", SmallInteger, nullable=False, server_default=text("0")),
    Column("runs_extras", SmallInteger, nullable=False, server_default=text("0")),
    Column("runs_total", SmallInteger, nullable=False, server_default=text("0")),

    # Extras breakdown
    Column("extras_type", String(20)),    # 'wide','no_ball','bye','leg_bye','penalty',NULL

    # Wickets
    Column("is_wicket", Boolean, nullable=False, server_default=text("false")),
    Column("wicket_kind", String(40)),    # 'caught','bowled','run out','lbw','stumped',...
    Column("player_out_id", Integer, ForeignKey("players.id")),

    # Derived fields (computed at ingestion — avoid re-computing on every query)
    Column("phase", String(20), nullable=False),                  # 'powerplay','middle','death'
    Column("is_boundary", Boolean, nullable=False, server_default=text("false")),
    Column("is_six", Boolean, nullable=False, server_default=text("false")),
    Column("is_dot_ball", Boolean, nullable=False, server_default=text("false")),
    Column("is_legal_delivery", Boolean, nullable=False, server_default=text("true")),
    Column("cumulative_runs", Integer),
    Column("cumulative_wickets", SmallInteger),

    UniqueConstraint("innings_id", "over_number", "ball_in_over",
                     name="uq_deliveries_position"),
)

dismissals = Table(
    "dismissals",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("delivery_id", BigInteger, ForeignKey("deliveries.id"), nullable=False),
    Column("player_out_id", Integer, ForeignKey("players.id"), nullable=False),
    Column("kind", String(40), nullable=False),
    Column("bowler_id", Integer, ForeignKey("players.id")),     # NULL for run outs
    Column("fielder1_id", Integer, ForeignKey("players.id")),
    Column("fielder2_id", Integer, ForeignKey("players.id")),
)

# ── Pre-computed Statistics ────────────────────────────────────────────────────

player_match_stats = Table(
    "player_match_stats",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("match_id", Integer, ForeignKey("matches.id"), nullable=False),
    Column("player_id", Integer, ForeignKey("players.id"), nullable=False),
    Column("team_id", Integer, ForeignKey("teams.id")),

    # Batting stats
    Column("runs_scored", Integer, server_default=text("0")),
    Column("balls_faced", Integer, server_default=text("0")),
    Column("fours", Integer, server_default=text("0")),
    Column("sixes", Integer, server_default=text("0")),
    Column("dot_balls_faced", Integer, server_default=text("0")),
    Column("batting_sr", Numeric(6, 2)),                # Strike rate (runs/balls*100)
    Column("batting_dismissed", Boolean, server_default=text("false")),

    # Bowling stats
    Column("overs_bowled", Numeric(4, 1), server_default=text("0")),
    Column("runs_conceded", Integer, server_default=text("0")),
    Column("wickets_taken", Integer, server_default=text("0")),
    Column("dot_balls_bowled", Integer, server_default=text("0")),
    Column("economy_rate", Numeric(5, 2)),               # runs_conceded / overs_bowled

    UniqueConstraint("match_id", "player_id", name="uq_player_match_stats"),
)

# ── AI Audit Trail ────────────────────────────────────────────────────────────

analysis_runs = Table(
    "analysis_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("query_text", Text, nullable=False),
    Column("match_id", Integer, ForeignKey("matches.id")),      # Optional context
    Column("player_id", Integer, ForeignKey("players.id")),     # Optional context
    Column("tools_called", JSONB),                               # List of tool names + results
    Column("llm_response", Text),
    Column("model_used", String(100)),
    Column("created_at", TIMESTAMP, server_default=text("NOW()")),
)

# ── Index Definitions ─────────────────────────────────────────────────────────
# Indexes are created here and applied by the migration.
# The deliveries table is the hot path for all analytics — heavily indexed.

INDEXES = [
    # deliveries — most queried table
    Index("idx_deliveries_innings",       deliveries.c.innings_id),
    Index("idx_deliveries_match",         deliveries.c.match_id),
    Index("idx_deliveries_batter",        deliveries.c.batter_id),
    Index("idx_deliveries_bowler",        deliveries.c.bowler_id),
    Index("idx_deliveries_batter_bowler", deliveries.c.batter_id, deliveries.c.bowler_id),
    Index("idx_deliveries_phase",         deliveries.c.phase),
    Index("idx_deliveries_match_phase",   deliveries.c.match_id, deliveries.c.phase),
    Index("idx_deliveries_legal",         deliveries.c.is_legal_delivery),

    # matches — filtered by competition, season, date
    Index("idx_matches_competition",      matches.c.competition_id),
    Index("idx_matches_season",           matches.c.season_id),
    Index("idx_matches_date",             matches.c.match_date),
    Index("idx_matches_team1",            matches.c.team1_id),
    Index("idx_matches_team2",            matches.c.team2_id),

    # player_match_stats — queried by player
    Index("idx_pms_player",               player_match_stats.c.player_id),
    Index("idx_pms_match",                player_match_stats.c.match_id),
]

# ── Materialized View SQL ─────────────────────────────────────────────────────
# Created manually (not via Alembic) — refresh with scripts/refresh_views.py

BATTER_CAREER_VIEW_SQL = """
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_batter_career AS
SELECT
    d.batter_id,
    COUNT(DISTINCT d.match_id)                                        AS matches,
    SUM(d.runs_batter)                                                AS total_runs,
    COUNT(*) FILTER (WHERE d.is_legal_delivery)                       AS balls_faced,
    MAX(pms.runs_scored)                                              AS highest_score,

    ROUND(
        SUM(d.runs_batter)::numeric /
        NULLIF(COUNT(*) FILTER (WHERE d.is_wicket AND d.player_out_id = d.batter_id), 0),
    2)                                                                AS average,

    ROUND(
        SUM(d.runs_batter)::numeric * 100 /
        NULLIF(COUNT(*) FILTER (WHERE d.is_legal_delivery), 0),
    2)                                                                AS strike_rate,

    ROUND(
        COUNT(*) FILTER (WHERE d.is_dot_ball)::numeric * 100 /
        NULLIF(COUNT(*) FILTER (WHERE d.is_legal_delivery), 0),
    1)                                                                AS dot_pct,

    ROUND(
        COUNT(*) FILTER (WHERE d.is_boundary)::numeric * 100 /
        NULLIF(COUNT(*) FILTER (WHERE d.is_legal_delivery), 0),
    1)                                                                AS boundary_pct,

    COUNT(*) FILTER (WHERE d.is_six)                                  AS total_sixes,
    COUNT(*) FILTER (WHERE d.is_boundary AND NOT d.is_six)            AS total_fours

FROM deliveries d
LEFT JOIN player_match_stats pms
       ON pms.player_id = d.batter_id AND pms.match_id = d.match_id
GROUP BY d.batter_id;
"""

BOWLER_CAREER_VIEW_SQL = """
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_bowler_career AS
SELECT
    d.bowler_id,
    COUNT(DISTINCT d.match_id)                                        AS matches,
    COUNT(*) FILTER (WHERE d.is_legal_delivery)                       AS balls_bowled,
    SUM(d.runs_total) FILTER (WHERE d.is_legal_delivery)              AS runs_conceded,
    COUNT(*) FILTER (WHERE d.is_wicket)                               AS wickets,

    ROUND(
        SUM(d.runs_total) FILTER (WHERE d.is_legal_delivery)::numeric * 6 /
        NULLIF(COUNT(*) FILTER (WHERE d.is_legal_delivery), 0),
    2)                                                                AS economy,

    ROUND(
        COUNT(*) FILTER (WHERE d.is_legal_delivery)::numeric /
        NULLIF(COUNT(*) FILTER (WHERE d.is_wicket), 0),
    2)                                                                AS bowling_sr,

    ROUND(
        COUNT(*) FILTER (WHERE d.is_dot_ball)::numeric * 100 /
        NULLIF(COUNT(*) FILTER (WHERE d.is_legal_delivery), 0),
    1)                                                                AS dot_pct

FROM deliveries d
GROUP BY d.bowler_id;
"""
