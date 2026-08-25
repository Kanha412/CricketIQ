"""
CricketIQ — Player Match Stats Computation
============================================
Computes per-match batting and bowling stats for every player
and populates the player_match_stats table.

Run this AFTER ingesting matches:
    python scripts/compute_player_stats.py
    python scripts/compute_player_stats.py --competition-id 1  (one competition only)

WHY A SEPARATE SCRIPT:
    Computing stats from raw deliveries with SQL is much faster than Python loops.
    This script runs two large SQL aggregations (one for batting, one for bowling)
    and merges them into player_match_stats.

    The player_match_stats table is used by Streamlit for fast player lookups
    instead of re-aggregating millions of deliveries on every page load.
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import get_transaction, test_connection
from src.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("scripts.compute_player_stats")
console = Console()


BATTING_STATS_SQL = """
INSERT INTO player_match_stats (
    match_id, player_id, team_id,
    runs_scored, balls_faced, fours, sixes,
    dot_balls_faced, batting_sr, batting_dismissed
)
SELECT
    d.match_id,
    d.batter_id                                                           AS player_id,
    i.batting_team_id                                                     AS team_id,

    SUM(d.runs_batter)                                                    AS runs_scored,
    COUNT(*) FILTER (WHERE d.is_legal_delivery)                           AS balls_faced,
    COUNT(*) FILTER (WHERE d.is_boundary AND NOT d.is_six)                AS fours,
    COUNT(*) FILTER (WHERE d.is_six)                                      AS sixes,
    COUNT(*) FILTER (WHERE d.is_dot_ball)                                 AS dot_balls_faced,

    ROUND(
        SUM(d.runs_batter)::numeric * 100 /
        NULLIF(COUNT(*) FILTER (WHERE d.is_legal_delivery), 0)
    , 2)                                                                  AS batting_sr,

    BOOL_OR(d.is_wicket AND d.player_out_id = d.batter_id)               AS batting_dismissed

FROM deliveries d
JOIN innings i ON i.id = d.innings_id
WHERE (:comp_id IS NULL OR d.match_id IN (
    SELECT id FROM matches WHERE competition_id = :comp_id
))
GROUP BY d.match_id, d.batter_id, i.batting_team_id

ON CONFLICT (match_id, player_id) DO UPDATE SET
    team_id         = EXCLUDED.team_id,
    runs_scored     = EXCLUDED.runs_scored,
    balls_faced     = EXCLUDED.balls_faced,
    fours           = EXCLUDED.fours,
    sixes           = EXCLUDED.sixes,
    dot_balls_faced = EXCLUDED.dot_balls_faced,
    batting_sr      = EXCLUDED.batting_sr,
    batting_dismissed = EXCLUDED.batting_dismissed
"""

BOWLING_STATS_SQL = """
INSERT INTO player_match_stats (
    match_id, player_id, team_id,
    overs_bowled, runs_conceded, wickets_taken, dot_balls_bowled, economy_rate
)
SELECT
    d.match_id,
    d.bowler_id                                                           AS player_id,
    i.bowling_team_id                                                     AS team_id,

    ROUND(
        COUNT(*) FILTER (WHERE d.is_legal_delivery)::numeric / 6
    , 1)                                                                  AS overs_bowled,

    SUM(d.runs_total) FILTER (WHERE d.is_legal_delivery)                  AS runs_conceded,
    COUNT(*) FILTER (WHERE d.is_wicket)                                   AS wickets_taken,
    COUNT(*) FILTER (WHERE d.is_dot_ball)                                 AS dot_balls_bowled,

    ROUND(
        SUM(d.runs_total) FILTER (WHERE d.is_legal_delivery)::numeric * 6 /
        NULLIF(COUNT(*) FILTER (WHERE d.is_legal_delivery), 0)
    , 2)                                                                  AS economy_rate

FROM deliveries d
JOIN innings i ON i.id = d.innings_id
WHERE (:comp_id IS NULL OR d.match_id IN (
    SELECT id FROM matches WHERE competition_id = :comp_id
))
GROUP BY d.match_id, d.bowler_id, i.bowling_team_id

ON CONFLICT (match_id, player_id) DO UPDATE SET
    overs_bowled    = EXCLUDED.overs_bowled,
    runs_conceded   = EXCLUDED.runs_conceded,
    wickets_taken   = EXCLUDED.wickets_taken,
    dot_balls_bowled = EXCLUDED.dot_balls_bowled,
    economy_rate    = EXCLUDED.economy_rate
"""


@click.command()
@click.option(
    "--competition-id",
    default=None,
    type=int,
    help="Compute only for a specific competition ID (default: all competitions)",
)
def main(competition_id: int | None) -> None:
    """
    Compute player_match_stats from the deliveries table.
    Run this after ingesting matches.
    """
    console.print("\n[bold]CricketIQ — Player Stats Computation[/bold]")

    if not test_connection():
        console.print("[red]Database connection failed.[/red]")
        sys.exit(1)

    console.print("Computing batting statistics...", end=" ")
    with get_transaction() as conn:
        result = conn.execute(text(BATTING_STATS_SQL), {"comp_id": competition_id})
    console.print(f"[green]Done[/green] ({result.rowcount} rows upserted)")

    console.print("Computing bowling statistics...", end=" ")
    with get_transaction() as conn:
        result = conn.execute(text(BOWLING_STATS_SQL), {"comp_id": competition_id})
    console.print(f"[green]Done[/green] ({result.rowcount} rows upserted)")

    # Summary count
    with get_transaction() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM player_match_stats")).scalar()
    console.print(f"\n[green]✓[/green] player_match_stats now has [bold]{count:,}[/bold] rows.")
    console.print("[dim]Player stats are ready for analytics and Streamlit dashboards.[/dim]\n")


if __name__ == "__main__":
    main()
