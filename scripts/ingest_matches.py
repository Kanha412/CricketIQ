"""
CricketIQ — Match Ingestion Pipeline
======================================
Reads parsed Cricsheet match data and inserts it into PostgreSQL.

USAGE:
    python scripts/ingest_matches.py --competitions ipl
    python scripts/ingest_matches.py --competitions ipl bbl --incremental
    python scripts/ingest_matches.py --competitions ipl --limit 10   (test with 10 files)

INSERTION ORDER (respects FK dependencies):
    1. competition (upsert)
    2. venue (upsert)
    3. teams (upsert, both teams)
    4. players (upsert, all from registry)
    5. season (upsert)
    6. match (insert)
    7. innings (insert, 2 per match)
    8. deliveries (bulk insert, ~250 per match)
    9. dismissals (insert)

DESIGN DECISIONS:
    - Uses "upsert" (INSERT ON CONFLICT DO NOTHING) for reference data
      so re-running is safe and idempotent
    - Deliveries use bulk INSERT (executemany) for performance
    - Each match is wrapped in a transaction — bad match = rollback that match only
    - Incremental mode checks DB for existing cricsheet_file_ids (source of truth)
    - Progress bar via rich for visibility on long ingestion runs
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.data.parsers import CricsheetParser, ParsedDelivery, ParsedInnings, ParsedMatch
from src.data.schema_models import CricsheetMatch
from src.data.validators import has_blocking_errors, validate_parsed_match
from src.db.connection import get_engine, get_transaction, test_connection
from src.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("scripts.ingest_matches")
console = Console()

parser = CricsheetParser()


# ── Upsert Helpers (idempotent inserts) ───────────────────────────────────────

def upsert_competition(conn, parsed: ParsedMatch) -> int:
    """
    Get or create a competition record.
    Returns the database ID.
    """
    # Determine match_type from the parsed match
    result = conn.execute(
        text("""
            INSERT INTO competitions (name, match_type, gender)
            VALUES (:name, :match_type, :gender)
            ON CONFLICT (name) DO UPDATE SET
                match_type = EXCLUDED.match_type,
                gender     = EXCLUDED.gender
            RETURNING id
        """),
        {
            "name": parsed.competition,
            "match_type": parsed.match_type,
            "gender": parsed.gender,
        },
    )
    return result.scalar_one()


def upsert_season(conn, competition_id: int, season: str) -> int:
    """Get or create a season. Returns DB ID."""
    result = conn.execute(
        text("""
            INSERT INTO seasons (competition_id, season)
            VALUES (:comp_id, :season)
            ON CONFLICT (competition_id, season) DO UPDATE SET season = EXCLUDED.season
            RETURNING id
        """),
        {"comp_id": competition_id, "season": season},
    )
    return result.scalar_one()


def upsert_venue(conn, parsed: ParsedMatch) -> int:
    """Get or create a venue. Returns DB ID."""
    result = conn.execute(
        text("""
            INSERT INTO venues (name, city)
            VALUES (:name, :city)
            ON CONFLICT (name) DO UPDATE SET city = COALESCE(EXCLUDED.city, venues.city)
            RETURNING id
        """),
        {"name": parsed.venue, "city": parsed.city},
    )
    return result.scalar_one()


def upsert_team(conn, team_name: str) -> int:
    """Get or create a team. Returns DB ID."""
    result = conn.execute(
        text("""
            INSERT INTO teams (name)
            VALUES (:name)
            ON CONFLICT (name) DO NOTHING
            RETURNING id
        """),
        {"name": team_name},
    )
    row = result.fetchone()
    if row:
        return row[0]
    # Already exists — fetch ID
    result = conn.execute(
        text("SELECT id FROM teams WHERE name = :name"),
        {"name": team_name},
    )
    return result.scalar_one()


def upsert_players(conn, registry: dict[str, str]) -> dict[str, int]:
    """
    Insert all players from the match registry.

    Args:
        registry: {player_name: cricsheet_id} from match info.

    Returns:
        {cricsheet_id: db_integer_id} mapping for use in delivery insertion.
    """
    id_map: dict[str, int] = {}

    for name, cricsheet_id in registry.items():
        # Try to insert — skip if cricsheet_id already exists
        result = conn.execute(
            text("""
                INSERT INTO players (cricsheet_id, name)
                VALUES (:cid, :name)
                ON CONFLICT (cricsheet_id) DO NOTHING
                RETURNING id
            """),
            {"cid": cricsheet_id, "name": name},
        )
        row = result.fetchone()
        if row:
            db_id = row[0]
        else:
            # Already exists — look up by cricsheet_id
            result = conn.execute(
                text("SELECT id FROM players WHERE cricsheet_id = :cid"),
                {"cid": cricsheet_id},
            )
            db_id = result.scalar_one()
        id_map[cricsheet_id] = db_id

    return id_map


def resolve_player_db_id(
    player_identifier: str,   # could be cricsheet_id or name (fallback)
    id_map: dict[str, int],   # cricsheet_id → db_id
    conn,
) -> int | None:
    """
    Resolve a player identifier (cricsheet_id or name) to a database integer ID.
    Returns None if player cannot be resolved.
    """
    # Direct lookup by cricsheet_id
    if player_identifier in id_map:
        return id_map[player_identifier]

    # Fallback: player not in registry — look up by name
    result = conn.execute(
        text("SELECT id FROM players WHERE name = :name LIMIT 1"),
        {"name": player_identifier},
    )
    row = result.fetchone()
    if row:
        return row[0]

    # Last resort: insert with name as both ID and name
    result = conn.execute(
        text("""
            INSERT INTO players (cricsheet_id, name)
            VALUES (:cid, :name)
            ON CONFLICT (cricsheet_id) DO NOTHING
            RETURNING id
        """),
        {"cid": player_identifier, "name": player_identifier},
    )
    row = result.fetchone()
    if row:
        return row[0]

    # Truly can't resolve
    logger.warning("Cannot resolve player: '%s'", player_identifier)
    return None


# ── Match Insertion ───────────────────────────────────────────────────────────

def insert_match(
    conn,
    parsed: ParsedMatch,
    competition_id: int,
    season_id: int,
    venue_id: int,
    team1_id: int,
    team2_id: int,
    toss_winner_id: int,
    winner_id: int | None,
    player_of_match_id: int | None,
    raw_json: dict,
) -> int:
    """Insert the match row. Returns match DB ID."""
    result = conn.execute(
        text("""
            INSERT INTO matches (
                cricsheet_file_id, competition_id, season_id, venue_id,
                match_date, match_type, gender,
                team1_id, team2_id, toss_winner_id, toss_decision,
                winner_id, win_margin_runs, win_margin_wickets, result, dls_applied,
                event_name, match_number, stage,
                player_of_match_id, raw_json
            ) VALUES (
                :file_id, :comp_id, :season_id, :venue_id,
                :match_date, :match_type, :gender,
                :team1_id, :team2_id, :toss_winner_id, :toss_decision,
                :winner_id, :win_runs, :win_wkts, :result, :dls,
                :event_name, :match_num, :stage,
                :pom_id, :raw_json
            ) RETURNING id
        """),
        {
            "file_id": parsed.cricsheet_file_id,
            "comp_id": competition_id,
            "season_id": season_id,
            "venue_id": venue_id,
            "match_date": parsed.match_date,
            "match_type": parsed.match_type,
            "gender": parsed.gender,
            "team1_id": team1_id,
            "team2_id": team2_id,
            "toss_winner_id": toss_winner_id,
            "toss_decision": parsed.toss_decision,
            "winner_id": winner_id,
            "win_runs": parsed.win_by_runs,
            "win_wkts": parsed.win_by_wickets,
            "result": parsed.result,
            "dls": parsed.dls_applied,
            "event_name": parsed.event_name,
            "match_num": parsed.match_number,
            "stage": parsed.stage,
            "pom_id": player_of_match_id,
            "raw_json": json.dumps(raw_json),
        },
    )
    return result.scalar_one()


def insert_innings(
    conn,
    match_id: int,
    inn: ParsedInnings,
    batting_team_id: int,
    bowling_team_id: int,
) -> int:
    """Insert an innings row. Returns innings DB ID."""
    # Compute total_overs from legal deliveries
    legal_count = sum(1 for d in inn.deliveries if d.is_legal_delivery)
    full_overs = legal_count // 6
    remainder = legal_count % 6
    total_overs = float(f"{full_overs}.{remainder}")

    result = conn.execute(
        text("""
            INSERT INTO innings (
                match_id, innings_number, batting_team_id, bowling_team_id,
                target_runs, total_runs, total_wickets, total_overs, is_super_over
            ) VALUES (
                :match_id, :inn_num, :bat_team, :bowl_team,
                :target, :runs, :wkts, :overs, :super_over
            ) RETURNING id
        """),
        {
            "match_id": match_id,
            "inn_num": inn.innings_number,
            "bat_team": batting_team_id,
            "bowl_team": bowling_team_id,
            "target": inn.target_runs,
            "runs": inn.total_runs,
            "wkts": inn.total_wickets,
            "overs": total_overs,
            "super_over": inn.is_super_over,
        },
    )
    return result.scalar_one()


def insert_deliveries_bulk(
    conn,
    match_id: int,
    innings_id: int,
    deliveries: list[ParsedDelivery],
    id_map: dict[str, int],
) -> list[tuple[int, ParsedDelivery]]:
    """
    Bulk-insert all deliveries for an innings.
    Returns list of (delivery_db_id, ParsedDelivery) for dismissal insertion.

    Uses executemany for performance (~10x faster than individual inserts).
    """
    rows = []
    for d in deliveries:
        batter_id   = resolve_player_db_id(d.batter_id,       id_map, conn)
        bowler_id   = resolve_player_db_id(d.bowler_id,       id_map, conn)
        ns_id       = resolve_player_db_id(d.non_striker_id,  id_map, conn)
        out_id      = resolve_player_db_id(d.player_out_id,   id_map, conn) if d.player_out_id else None

        if batter_id is None or bowler_id is None:
            logger.warning(
                "Skipping delivery — cannot resolve batter=%s or bowler=%s",
                d.batter_id, d.bowler_id
            )
            continue

        rows.append({
            "innings_id":       innings_id,
            "match_id":         match_id,
            "over_number":      d.over_number,
            "ball_in_over":     d.ball_in_over,
            "legal_ball_num":   d.legal_ball_number,
            "batter_id":        batter_id,
            "non_striker_id":   ns_id,
            "bowler_id":        bowler_id,
            "runs_batter":      d.runs_batter,
            "runs_extras":      d.runs_extras,
            "runs_total":       d.runs_total,
            "extras_type":      d.extras_type,
            "is_wicket":        d.is_wicket,
            "wicket_kind":      d.wicket_kind,
            "player_out_id":    out_id,
            "phase":            d.phase,
            "is_boundary":      d.is_boundary,
            "is_six":           d.is_six,
            "is_dot_ball":      d.is_dot_ball,
            "is_legal":         d.is_legal_delivery,
            "cum_runs":         d.cumulative_runs,
            "cum_wkts":         d.cumulative_wickets,
        })

    if not rows:
        return []

    # Bulk insert — executemany does NOT support RETURNING in psycopg2/SQLAlchemy,
    # so we omit RETURNING here and look up IDs separately for wicket rows only.
    conn.execute(
        text("""
            INSERT INTO deliveries (
                innings_id, match_id, over_number, ball_in_over, legal_ball_number,
                batter_id, non_striker_id, bowler_id,
                runs_batter, runs_extras, runs_total, extras_type,
                is_wicket, wicket_kind, player_out_id,
                phase, is_boundary, is_six, is_dot_ball, is_legal_delivery,
                cumulative_runs, cumulative_wickets
            ) VALUES (
                :innings_id, :match_id, :over_number, :ball_in_over, :legal_ball_num,
                :batter_id, :non_striker_id, :bowler_id,
                :runs_batter, :runs_extras, :runs_total, :extras_type,
                :is_wicket, :wicket_kind, :player_out_id,
                :phase, :is_boundary, :is_six, :is_dot_ball, :is_legal,
                :cum_runs, :cum_wkts
            )
            ON CONFLICT (innings_id, over_number, ball_in_over) DO NOTHING
        """),
        rows,
    )

    # For wicket deliveries, look up their DB IDs via the unique key so we can
    # insert dismissal records that reference them.
    wicket_deliveries: list[tuple[int, ParsedDelivery]] = []
    for d in deliveries:
        if not d.is_wicket:
            continue
        row = conn.execute(
            text("""
                SELECT id FROM deliveries
                WHERE innings_id = :iid
                  AND over_number = :over
                  AND ball_in_over = :ball
            """),
            {"iid": innings_id, "over": d.over_number, "ball": d.ball_in_over},
        ).fetchone()
        if row:
            wicket_deliveries.append((row[0], d))
    return wicket_deliveries


def insert_dismissals(
    conn,
    delivery_id: int,
    delivery: ParsedDelivery,
    id_map: dict[str, int],
) -> None:
    """Insert dismissal record(s) for a wicket delivery."""
    if not delivery.is_wicket:
        return

    out_id = resolve_player_db_id(delivery.player_out_id, id_map, conn) if delivery.player_out_id else None
    bowler_id = resolve_player_db_id(delivery.bowler_id, id_map, conn)

    fielder_ids = []
    for fname in delivery.fielder_names[:2]:    # max 2 fielders
        fid = resolve_player_db_id(fname, id_map, conn)
        fielder_ids.append(fid)

    conn.execute(
        text("""
            INSERT INTO dismissals (delivery_id, player_out_id, kind, bowler_id, fielder1_id, fielder2_id)
            VALUES (:del_id, :out_id, :kind, :bowler_id, :f1, :f2)
        """),
        {
            "del_id":   delivery_id,
            "out_id":   out_id,
            "kind":     delivery.wicket_kind or "unknown",
            "bowler_id": bowler_id,
            "f1":       fielder_ids[0] if len(fielder_ids) > 0 else None,
            "f2":       fielder_ids[1] if len(fielder_ids) > 1 else None,
        },
    )


# ── Full Match Ingestion ──────────────────────────────────────────────────────

def ingest_match(parsed: ParsedMatch, raw_json: dict) -> bool:
    """
    Insert a single parsed match into the database.
    Wrapped in a transaction — failure rolls back this match only.

    Returns:
        True on success, False on failure.
    """
    try:
        with get_transaction() as conn:

            # 1. Reference data (all upserts — safe to call repeatedly)
            competition_id = upsert_competition(conn, parsed)
            season_id      = upsert_season(conn, competition_id, parsed.season)
            venue_id       = upsert_venue(conn, parsed)
            team1_id       = upsert_team(conn, parsed.team1)
            team2_id       = upsert_team(conn, parsed.team2)
            toss_team_id   = upsert_team(conn, parsed.toss_winner)

            # 2. Winner team ID (None for ties/no-result)
            winner_id = None
            if parsed.winner:
                winner_id = upsert_team(conn, parsed.winner)

            # 3. Players — build cricsheet_id → db_id map
            id_map = upsert_players(conn, parsed.player_registry)

            # 4. Player of match (take first if multiple)
            pom_id = None
            if parsed.player_of_match:
                pom_name = parsed.player_of_match[0]
                pom_cricsheet_id = parsed.player_registry.get(pom_name, pom_name)
                pom_id = resolve_player_db_id(pom_cricsheet_id, id_map, conn)

            # 5. Insert match row
            match_id = insert_match(
                conn, parsed,
                competition_id, season_id, venue_id,
                team1_id, team2_id, toss_team_id,
                winner_id, pom_id, raw_json,
            )

            # 6. Insert innings + deliveries
            for inn in parsed.innings:
                # Determine batting/bowling team IDs
                bat_id  = team1_id if inn.batting_team == parsed.team1 else team2_id
                bowl_id = team2_id if inn.batting_team == parsed.team1 else team1_id

                innings_id = insert_innings(conn, match_id, inn, bat_id, bowl_id)

                # Bulk insert deliveries
                wicket_deliveries = insert_deliveries_bulk(
                    conn, match_id, innings_id, inn.deliveries, id_map
                )

                # Insert dismissals for wicket deliveries
                for del_id, del_obj in wicket_deliveries:
                    insert_dismissals(conn, del_id, del_obj, id_map)

        return True

    except Exception as e:
        logger.error("Failed to ingest match %s: %s", parsed.cricsheet_file_id, e, exc_info=True)
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--competitions", "-c",
    multiple=True,
    default=["ipl"],
    help="Competitions to ingest",
    show_default=True,
)
@click.option(
    "--incremental/--no-incremental",
    default=True,
    help="Skip matches already in DB",
    show_default=True,
)
@click.option(
    "--limit",
    default=0,
    type=int,
    help="Limit to first N matches per competition (0 = no limit). Useful for testing.",
)
def main(competitions: tuple[str, ...], incremental: bool, limit: int) -> None:
    """
    Ingest Cricsheet JSON match data into PostgreSQL.

    Reads from data/raw/cricsheet/{competition}/ and inserts into the database.
    Safe to re-run — skips matches already in DB when --incremental is set.
    """
    # Verify DB connection
    console.print("\n[bold]CricketIQ — Match Ingestion Pipeline[/bold]")
    console.print(f"Competitions : [cyan]{', '.join(competitions)}[/cyan]")
    console.print(f"Mode         : [cyan]{'incremental' if incremental else 'full re-ingest'}[/cyan]")
    console.print(f"Limit        : [cyan]{'None' if limit == 0 else limit}[/cyan]\n")

    console.print("Testing database connection...", end=" ")
    if not test_connection():
        console.print("[red]FAILED[/red]")
        console.print("[red]Check DATABASE_URL in your .env file and ensure PostgreSQL is running.[/red]")
        sys.exit(1)
    console.print("[green]OK[/green]\n")

    # Get already-ingested match IDs (incremental mode)
    from src.db.queries import get_ingested_file_ids
    existing_ids: set[str] = get_ingested_file_ids() if incremental else set()
    if existing_ids:
        console.print(f"[dim]Found {len(existing_ids)} already-ingested matches — will skip.[/dim]\n")

    total_ingested = 0
    total_skipped = 0
    total_failed = 0

    for competition in competitions:
        raw_dir = settings.data_raw_dir / "cricsheet" / competition
        if not raw_dir.exists():
            console.print(f"[yellow]![/yellow] {competition}: No data found at {raw_dir}")
            console.print(f"   Run: python scripts/download_cricsheet.py --competitions {competition}")
            continue

        all_files = sorted(raw_dir.glob("*.json"))
        new_files = [f for f in all_files if f.stem not in existing_ids]

        if limit > 0:
            new_files = new_files[:limit]

        skipped = len(all_files) - len(new_files)
        console.print(
            f"[cyan]{competition.upper()}[/cyan]: "
            f"{len(all_files)} total, {len(new_files)} to ingest, {skipped} skipped"
        )

        if not new_files:
            console.print(f"  [dim]All {competition} matches already ingested.[/dim]")
            total_skipped += skipped
            continue

        # Ingest with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold]{competition}[/bold]"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("remaining:"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("ingest", total=len(new_files))
            ingested = 0
            failed = 0

            for match_file in new_files:
                try:
                    # Load raw JSON (kept for storage in matches.raw_json)
                    with open(match_file, encoding="utf-8") as f:
                        raw_json = json.load(f)

                    # Parse
                    cricsheet_match = CricsheetMatch.model_validate(raw_json)
                    parsed = parser.parse(cricsheet_match, file_id=match_file.stem)

                    # Validate
                    issues = validate_parsed_match(parsed)
                    if has_blocking_errors(issues):
                        logger.error("Skipping %s — validation errors", match_file.stem)
                        failed += 1
                        progress.advance(task)
                        continue

                    # Ingest
                    success = ingest_match(parsed, raw_json)
                    if success:
                        ingested += 1
                    else:
                        failed += 1

                except Exception as e:
                    logger.error("Unexpected error on %s: %s", match_file.stem, e, exc_info=False)
                    failed += 1

                progress.advance(task)

        console.print(
            f"  [green]✓[/green] {ingested} ingested, "
            f"{skipped} skipped, "
            f"{'[red]' if failed > 0 else ''}{failed} failed{'[/red]' if failed > 0 else ''}"
        )
        total_ingested += ingested
        total_skipped += skipped
        total_failed += failed

    # Summary
    console.print(f"\n[bold]Ingestion complete.[/bold]")
    console.print(f"  Ingested : {total_ingested}")
    console.print(f"  Skipped  : {total_skipped}")
    console.print(f"  Failed   : {total_failed}")

    if total_ingested > 0:
        console.print(
            "\n[dim]Next: Compute player stats:[/dim]\n"
            "[bold]  python scripts/compute_player_stats.py[/bold]"
        )


if __name__ == "__main__":
    main()
