"""Initial schema — all tables, indexes, and constraints

Revision ID: 001
Revises: (none — first migration)
Create Date: 2026-08-25

Creates:
    - competitions, venues, teams, players      (reference tables)
    - seasons, matches, innings                  (match hierarchy)
    - deliveries, dismissals                     (ball-by-ball data)
    - player_match_stats                         (pre-computed stats)
    - analysis_runs                              (AI audit trail)
    - All indexes on deliveries and matches
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Reference Tables ──────────────────────────────────────────────────────

    op.create_table(
        "competitions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(150), nullable=False),
        sa.Column("short_name", sa.String(20)),
        sa.Column("gender", sa.String(10)),
        sa.Column("match_type", sa.String(20)),
        sa.UniqueConstraint("name", name="uq_competitions_name"),
    )

    op.create_table(
        "venues",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(250), nullable=False),
        sa.Column("city", sa.String(100)),
        sa.Column("country", sa.String(100)),
        sa.UniqueConstraint("name", name="uq_venues_name"),
    )

    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(150), nullable=False),
        sa.Column("short_name", sa.String(20)),
        sa.UniqueConstraint("name", name="uq_teams_name"),
    )

    op.create_table(
        "players",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("cricsheet_id", sa.String(50)),
        sa.Column("name", sa.String(150), nullable=False),
        sa.Column("batting_style", sa.String(60)),
        sa.Column("bowling_style", sa.String(60)),
        sa.Column("role", sa.String(50)),
        sa.Column("nationality", sa.String(100)),
        sa.UniqueConstraint("cricsheet_id", name="uq_players_cricsheet_id"),
    )

    # ── Match Tables ──────────────────────────────────────────────────────────

    op.create_table(
        "seasons",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("competition_id", sa.Integer(), sa.ForeignKey("competitions.id"), nullable=False),
        sa.Column("season", sa.String(20), nullable=False),
        sa.UniqueConstraint("competition_id", "season", name="uq_seasons_comp_season"),
    )

    op.create_table(
        "matches",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("cricsheet_file_id", sa.String(50), nullable=False),
        sa.Column("competition_id", sa.Integer(), sa.ForeignKey("competitions.id")),
        sa.Column("season_id", sa.Integer(), sa.ForeignKey("seasons.id")),
        sa.Column("venue_id", sa.Integer(), sa.ForeignKey("venues.id")),
        sa.Column("match_date", sa.Date(), nullable=False),
        sa.Column("match_type", sa.String(20), nullable=False),
        sa.Column("gender", sa.String(10)),
        sa.Column("team1_id", sa.Integer(), sa.ForeignKey("teams.id")),
        sa.Column("team2_id", sa.Integer(), sa.ForeignKey("teams.id")),
        sa.Column("toss_winner_id", sa.Integer(), sa.ForeignKey("teams.id")),
        sa.Column("toss_decision", sa.String(10)),
        sa.Column("winner_id", sa.Integer(), sa.ForeignKey("teams.id")),
        sa.Column("win_margin_runs", sa.Integer()),
        sa.Column("win_margin_wickets", sa.Integer()),
        sa.Column("result", sa.String(20)),
        sa.Column("dls_applied", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("event_name", sa.String(150)),
        sa.Column("match_number", sa.Integer()),
        sa.Column("stage", sa.String(60)),
        sa.Column("player_of_match_id", sa.Integer(), sa.ForeignKey("players.id")),
        sa.Column("raw_json", postgresql.JSONB()),
        sa.Column("ingested_at", postgresql.TIMESTAMP(), server_default=sa.text("NOW()")),
        sa.UniqueConstraint("cricsheet_file_id", name="uq_matches_file_id"),
    )

    op.create_table(
        "innings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id", ondelete="CASCADE"), nullable=False),
        sa.Column("innings_number", sa.SmallInteger(), nullable=False),
        sa.Column("batting_team_id", sa.Integer(), sa.ForeignKey("teams.id")),
        sa.Column("bowling_team_id", sa.Integer(), sa.ForeignKey("teams.id")),
        sa.Column("target_runs", sa.Integer()),
        sa.Column("total_runs", sa.Integer()),
        sa.Column("total_wickets", sa.SmallInteger()),
        sa.Column("total_overs", sa.Numeric(4, 1)),
        sa.Column("is_super_over", sa.Boolean(), server_default=sa.text("false")),
        sa.UniqueConstraint("match_id", "innings_number", name="uq_innings_match_innings"),
    )

    # ── Core Analytical Table ─────────────────────────────────────────────────

    op.create_table(
        "deliveries",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("innings_id", sa.Integer(), sa.ForeignKey("innings.id", ondelete="CASCADE"), nullable=False),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("over_number", sa.SmallInteger(), nullable=False),
        sa.Column("ball_in_over", sa.SmallInteger(), nullable=False),
        sa.Column("legal_ball_number", sa.SmallInteger()),
        sa.Column("batter_id", sa.Integer(), sa.ForeignKey("players.id"), nullable=False),
        sa.Column("non_striker_id", sa.Integer(), sa.ForeignKey("players.id")),
        sa.Column("bowler_id", sa.Integer(), sa.ForeignKey("players.id"), nullable=False),
        sa.Column("runs_batter", sa.SmallInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("runs_extras", sa.SmallInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("runs_total", sa.SmallInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("extras_type", sa.String(20)),
        sa.Column("is_wicket", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("wicket_kind", sa.String(40)),
        sa.Column("player_out_id", sa.Integer(), sa.ForeignKey("players.id")),
        sa.Column("phase", sa.String(20), nullable=False),
        sa.Column("is_boundary", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_six", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_dot_ball", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_legal_delivery", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("cumulative_runs", sa.Integer()),
        sa.Column("cumulative_wickets", sa.SmallInteger()),
        sa.UniqueConstraint("innings_id", "over_number", "ball_in_over",
                            name="uq_deliveries_position"),
    )

    op.create_table(
        "dismissals",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("delivery_id", sa.BigInteger(), sa.ForeignKey("deliveries.id"), nullable=False),
        sa.Column("player_out_id", sa.Integer(), sa.ForeignKey("players.id"), nullable=False),
        sa.Column("kind", sa.String(40), nullable=False),
        sa.Column("bowler_id", sa.Integer(), sa.ForeignKey("players.id")),
        sa.Column("fielder1_id", sa.Integer(), sa.ForeignKey("players.id")),
        sa.Column("fielder2_id", sa.Integer(), sa.ForeignKey("players.id")),
    )

    # ── Pre-computed Stats ────────────────────────────────────────────────────

    op.create_table(
        "player_match_stats",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("player_id", sa.Integer(), sa.ForeignKey("players.id"), nullable=False),
        sa.Column("team_id", sa.Integer(), sa.ForeignKey("teams.id")),
        sa.Column("runs_scored", sa.Integer(), server_default=sa.text("0")),
        sa.Column("balls_faced", sa.Integer(), server_default=sa.text("0")),
        sa.Column("fours", sa.Integer(), server_default=sa.text("0")),
        sa.Column("sixes", sa.Integer(), server_default=sa.text("0")),
        sa.Column("dot_balls_faced", sa.Integer(), server_default=sa.text("0")),
        sa.Column("batting_sr", sa.Numeric(6, 2)),
        sa.Column("batting_dismissed", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("overs_bowled", sa.Numeric(4, 1), server_default=sa.text("0")),
        sa.Column("runs_conceded", sa.Integer(), server_default=sa.text("0")),
        sa.Column("wickets_taken", sa.Integer(), server_default=sa.text("0")),
        sa.Column("dot_balls_bowled", sa.Integer(), server_default=sa.text("0")),
        sa.Column("economy_rate", sa.Numeric(5, 2)),
        sa.UniqueConstraint("match_id", "player_id", name="uq_player_match_stats"),
    )

    # ── AI Audit Trail ────────────────────────────────────────────────────────

    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id")),
        sa.Column("player_id", sa.Integer(), sa.ForeignKey("players.id")),
        sa.Column("tools_called", postgresql.JSONB()),
        sa.Column("llm_response", sa.Text()),
        sa.Column("model_used", sa.String(100)),
        sa.Column("created_at", postgresql.TIMESTAMP(), server_default=sa.text("NOW()")),
    )

    # ── Indexes ───────────────────────────────────────────────────────────────
    op.create_index("idx_deliveries_innings",       "deliveries", ["innings_id"])
    op.create_index("idx_deliveries_match",         "deliveries", ["match_id"])
    op.create_index("idx_deliveries_batter",        "deliveries", ["batter_id"])
    op.create_index("idx_deliveries_bowler",        "deliveries", ["bowler_id"])
    op.create_index("idx_deliveries_batter_bowler", "deliveries", ["batter_id", "bowler_id"])
    op.create_index("idx_deliveries_phase",         "deliveries", ["phase"])
    op.create_index("idx_deliveries_match_phase",   "deliveries", ["match_id", "phase"])
    op.create_index("idx_deliveries_legal",         "deliveries", ["is_legal_delivery"])
    op.create_index("idx_matches_competition",      "matches",    ["competition_id"])
    op.create_index("idx_matches_season",           "matches",    ["season_id"])
    op.create_index("idx_matches_date",             "matches",    ["match_date"])
    op.create_index("idx_matches_team1",            "matches",    ["team1_id"])
    op.create_index("idx_matches_team2",            "matches",    ["team2_id"])
    op.create_index("idx_pms_player",               "player_match_stats", ["player_id"])
    op.create_index("idx_pms_match",                "player_match_stats", ["match_id"])


def downgrade() -> None:
    # Drop in reverse order (respecting foreign key dependencies)
    op.drop_table("analysis_runs")
    op.drop_table("player_match_stats")
    op.drop_table("dismissals")
    op.drop_table("deliveries")
    op.drop_table("innings")
    op.drop_table("matches")
    op.drop_table("seasons")
    op.drop_table("players")
    op.drop_table("teams")
    op.drop_table("venues")
    op.drop_table("competitions")
