"""
CricketIQ — Fast Bulk Ingest for Cloud PostgreSQL (Neon)
=========================================================
Optimised for high-latency cloud connections.

PROBLEM with the normal ingest script on Neon:
    - executemany() in psycopg2 sends ONE SQL per row
    - 295,732 deliveries × 200ms latency = 16 HOURS

THIS SCRIPT:
    - Parses ALL JSON files first (pure Python, no DB)
    - Inserts ALL data in ~30 SQL calls using execute_values()
    - Expected time: 5–15 minutes instead of 16 hours

HOW:
    Phase 1  Parse all 1,243 JSON files in memory
    Phase 2  Bulk-upsert reference tables (competitions, seasons, venues, teams, players)
    Phase 3  Bulk-insert matches → get all IDs back in one go
    Phase 4  Bulk-insert innings → get all IDs back
    Phase 5  Bulk-insert deliveries with execute_values (page_size=2000)
    Phase 6  Bulk-insert dismissals

USAGE:
    DATABASE_URL=<neon_url> python scripts/fast_ingest_neon.py
    python scripts/fast_ingest_neon.py --limit 50   # test with 50 matches first
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import settings
from src.data.parsers import CricsheetParser, ParsedDelivery, ParsedMatch
from src.data.schema_models import CricsheetMatch
from src.data.validators import has_blocking_errors, validate_parsed_match
from src.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("scripts.fast_ingest_neon")


def parse_args():
    p = argparse.ArgumentParser(description="Fast bulk ingest to Neon PostgreSQL")
    p.add_argument("--limit",  type=int, default=0, help="Limit to first N matches (0=all)")
    p.add_argument("--competition", default="ipl", help="Competition folder name")
    p.add_argument("--db-url", default=None, help="Override DATABASE_URL")
    return p.parse_args()


# ── Phase 1: Parse all JSON files ────────────────────────────────────────────

def parse_all_files(raw_dir: Path, limit: int) -> tuple[list[ParsedMatch], dict]:
    """
    Parse all Cricsheet JSON files. Returns (parsed_matches, raw_json_map).
    raw_json_map: {file_stem: raw_json_dict}  (stored in matches.raw_json)
    """
    cricket_parser = CricsheetParser()
    json_files = sorted(raw_dir.glob("*.json"))
    if limit > 0:
        json_files = json_files[:limit]

    print(f"Parsing {len(json_files)} files...")
    parsed_matches = []
    raw_jsons = {}
    skipped = 0

    for i, f in enumerate(json_files, 1):
        try:
            with open(f, encoding="utf-8") as fh:
                raw = json.load(fh)
            cricsheet_match = CricsheetMatch.model_validate(raw)
            parsed = cricket_parser.parse(cricsheet_match, file_id=f.stem)
            issues = validate_parsed_match(parsed)
            if has_blocking_errors(issues):
                logger.warning("Skipping %s — validation errors", f.stem)
                skipped += 1
                continue
            parsed_matches.append(parsed)
            raw_jsons[f.stem] = raw
        except Exception as e:
            logger.error("Error parsing %s: %s", f.stem, e)
            skipped += 1

        if i % 100 == 0:
            print(f"  {i}/{len(json_files)} parsed...")

    print(f"Parsed {len(parsed_matches)} matches ({skipped} skipped)")
    return parsed_matches, raw_jsons


# ── Phase 2: Bulk-upsert reference tables ────────────────────────────────────

def load_reference_tables(cursor, parsed_matches: list[ParsedMatch]) -> dict:
    """
    Collect all unique reference data across all matches and bulk-upsert.
    Returns id_maps: {
        'comp':    {name: id},
        'season':  {(comp_name, season): id},
        'venue':   {name: id},
        'team':    {name: id},
        'player':  {cricsheet_id: db_id},
    }
    """
    # Collect unique values
    competitions: dict[str, dict] = {}
    seasons_meta: dict[tuple, dict] = {}
    venues: dict[str, dict] = {}
    teams: set[str] = set()
    players: dict[str, str] = {}   # cricsheet_id → name

    for p in parsed_matches:
        competitions[p.competition] = {
            "name": p.competition, "match_type": p.match_type, "gender": p.gender
        }
        seasons_meta[(p.competition, p.season)] = {
            "comp_name": p.competition, "season": p.season
        }
        venues[p.venue] = {"name": p.venue, "city": p.city}
        teams.update([p.team1, p.team2, p.toss_winner])
        if p.winner:
            teams.add(p.winner)
        for name, cid in p.player_registry.items():
            players[cid] = name

    print(f"  Reference: {len(competitions)} comps, {len(seasons_meta)} seasons, "
          f"{len(venues)} venues, {len(teams)} teams, {len(players)} players")

    # ── Competitions ─────────────────────────────────────────────────────────
    psycopg2.extras.execute_values(
        cursor,
        """INSERT INTO competitions (name, match_type, gender) VALUES %s
           ON CONFLICT (name) DO UPDATE SET match_type=EXCLUDED.match_type, gender=EXCLUDED.gender
           RETURNING id, name""",
        [(c["name"], c["match_type"], c["gender"]) for c in competitions.values()],
    )
    comp_map = {row[1]: row[0] for row in cursor.fetchall()}

    # ── Seasons ───────────────────────────────────────────────────────────────
    psycopg2.extras.execute_values(
        cursor,
        """INSERT INTO seasons (competition_id, season) VALUES %s
           ON CONFLICT (competition_id, season) DO UPDATE SET season=EXCLUDED.season
           RETURNING id, competition_id, season""",
        [(comp_map[s["comp_name"]], s["season"]) for s in seasons_meta.values()],
    )
    raw_seasons = cursor.fetchall()   # (id, comp_id, season_str)
    # Build (comp_name, season_str) → id using comp_id to comp_name reverse map
    comp_name_rev = {v: k for k, v in comp_map.items()}
    season_map = {
        (comp_name_rev[row[1]], row[2]): row[0]
        for row in raw_seasons
    }

    # ── Venues ────────────────────────────────────────────────────────────────
    psycopg2.extras.execute_values(
        cursor,
        """INSERT INTO venues (name, city) VALUES %s
           ON CONFLICT (name) DO UPDATE SET city=COALESCE(EXCLUDED.city, venues.city)
           RETURNING id, name""",
        [(v["name"], v["city"]) for v in venues.values()],
    )
    venue_map = {row[1]: row[0] for row in cursor.fetchall()}

    # ── Teams ─────────────────────────────────────────────────────────────────
    psycopg2.extras.execute_values(
        cursor,
        """INSERT INTO teams (name) VALUES %s
           ON CONFLICT (name) DO NOTHING
           RETURNING id, name""",
        [(t,) for t in sorted(teams)],
    )
    inserted_teams = {row[1]: row[0] for row in cursor.fetchall()}

    # Fetch ALL teams (including pre-existing ones not returned by DO NOTHING)
    cursor.execute("SELECT id, name FROM teams")
    team_map = {row[1]: row[0] for row in cursor.fetchall()}

    # ── Players ───────────────────────────────────────────────────────────────
    psycopg2.extras.execute_values(
        cursor,
        """INSERT INTO players (cricsheet_id, name) VALUES %s
           ON CONFLICT (cricsheet_id) DO NOTHING""",
        [(cid, name) for cid, name in players.items()],
        page_size=500,
    )
    # Fetch ALL players to build cricsheet_id → db_id map
    cursor.execute("SELECT id, cricsheet_id FROM players")
    player_map = {row[1]: row[0] for row in cursor.fetchall()}

    return {
        "comp":   comp_map,
        "season": season_map,
        "venue":  venue_map,
        "team":   team_map,
        "player": player_map,
    }


# ── Phase 3: Bulk-insert matches ──────────────────────────────────────────────

def insert_all_matches(
    cursor,
    parsed_matches: list[ParsedMatch],
    raw_jsons: dict,
    id_maps: dict,
) -> dict[str, int]:
    """
    Bulk-insert all matches. Returns {cricsheet_file_id: match_db_id}.
    """
    comp_map   = id_maps["comp"]
    season_map = id_maps["season"]
    venue_map  = id_maps["venue"]
    team_map   = id_maps["team"]
    player_map = id_maps["player"]

    rows = []
    for p in parsed_matches:
        winner_id  = team_map.get(p.winner)
        pom_id     = None
        if p.player_of_match:
            pom_cid = p.player_registry.get(p.player_of_match[0], p.player_of_match[0])
            pom_id  = player_map.get(pom_cid)

        rows.append((
            p.cricsheet_file_id,
            comp_map[p.competition],
            season_map[(p.competition, p.season)],
            venue_map[p.venue],
            p.match_date,
            p.match_type,
            p.gender,
            team_map[p.team1],
            team_map[p.team2],
            team_map[p.toss_winner],
            p.toss_decision,
            winner_id,
            p.win_by_runs,
            p.win_by_wickets,
            p.result,
            p.dls_applied,
            p.event_name,
            p.match_number,
            p.stage,
            pom_id,
            json.dumps(raw_jsons[p.cricsheet_file_id]),
        ))

    psycopg2.extras.execute_values(
        cursor,
        """INSERT INTO matches (
            cricsheet_file_id, competition_id, season_id, venue_id,
            match_date, match_type, gender,
            team1_id, team2_id, toss_winner_id, toss_decision,
            winner_id, win_margin_runs, win_margin_wickets, result, dls_applied,
            event_name, match_number, stage,
            player_of_match_id, raw_json
        ) VALUES %s
        ON CONFLICT (cricsheet_file_id) DO NOTHING
        RETURNING id, cricsheet_file_id""",
        rows,
        page_size=200,
    )
    match_db_ids = {row[1]: row[0] for row in cursor.fetchall()}

    # If some already existed (ON CONFLICT DO NOTHING), fetch their IDs
    if len(match_db_ids) < len(parsed_matches):
        cursor.execute("SELECT id, cricsheet_file_id FROM matches")
        for row in cursor.fetchall():
            if row[1] not in match_db_ids:
                match_db_ids[row[1]] = row[0]

    print(f"  Matches inserted/found: {len(match_db_ids)}")
    return match_db_ids


# ── Phase 4: Bulk-insert innings ──────────────────────────────────────────────

def insert_all_innings(
    cursor,
    parsed_matches: list[ParsedMatch],
    match_db_ids: dict[str, int],
    id_maps: dict,
) -> dict[tuple, int]:
    """
    Bulk-insert innings. Returns {(match_db_id, innings_number): innings_db_id}.
    """
    team_map = id_maps["team"]
    rows = []

    for p in parsed_matches:
        match_id = match_db_ids.get(p.cricsheet_file_id)
        if match_id is None:
            continue
        for inn in p.innings:
            bat_id  = team_map.get(inn.batting_team)
            bowl_id = team_map.get(inn.bowling_team) if hasattr(inn, 'bowling_team') else None

            # Derive bowling team if not stored
            if bowl_id is None:
                bowl_team = p.team2 if inn.batting_team == p.team1 else p.team1
                bowl_id = team_map.get(bowl_team)

            legal_count = sum(1 for d in inn.deliveries if d.is_legal_delivery)
            full_overs  = legal_count // 6
            remainder   = legal_count % 6
            total_overs = float(f"{full_overs}.{remainder}")

            rows.append((
                match_id,
                inn.innings_number,
                bat_id,
                bowl_id,
                inn.target_runs,
                inn.total_runs,
                inn.total_wickets,
                total_overs,
                inn.is_super_over,
            ))

    psycopg2.extras.execute_values(
        cursor,
        """INSERT INTO innings (
            match_id, innings_number, batting_team_id, bowling_team_id,
            target_runs, total_runs, total_wickets, total_overs, is_super_over
        ) VALUES %s
        ON CONFLICT DO NOTHING""",
        rows,
        page_size=1000,
    )
    # Always SELECT — execute_values RETURNING only returns the LAST page
    cursor.execute(
        "SELECT id, match_id, innings_number FROM innings WHERE match_id = ANY(%s)",
        (list(match_db_ids.values()),)
    )
    innings_map = {(row[1], row[2]): row[0] for row in cursor.fetchall()}

    print(f"  Innings inserted/found: {len(innings_map)}")
    return innings_map


# ── Phase 5: Bulk-insert deliveries ───────────────────────────────────────────

def insert_all_deliveries(
    cursor,
    parsed_matches: list[ParsedMatch],
    match_db_ids: dict[str, int],
    innings_map: dict[tuple, int],
    id_maps: dict,
) -> dict[tuple, int]:
    """
    Bulk-insert ALL deliveries using execute_values.
    Returns {(innings_id, over_number, ball_in_over): delivery_db_id}
    for wicket rows only (used for dismissal insertion).
    """
    player_map = id_maps["player"]
    all_rows = []

    for p in parsed_matches:
        match_id = match_db_ids.get(p.cricsheet_file_id)
        if match_id is None:
            continue
        for inn in p.innings:
            innings_id = innings_map.get((match_id, inn.innings_number))
            if innings_id is None:
                continue
            for d in inn.deliveries:
                batter_id = player_map.get(d.batter_id)
                bowler_id = player_map.get(d.bowler_id)
                ns_id     = player_map.get(d.non_striker_id)
                out_id    = player_map.get(d.player_out_id) if d.player_out_id else None

                if batter_id is None or bowler_id is None:
                    continue

                all_rows.append((
                    innings_id,
                    match_id,
                    d.over_number,
                    d.ball_in_over,
                    d.legal_ball_number,
                    batter_id,
                    ns_id,
                    bowler_id,
                    d.runs_batter,
                    d.runs_extras,
                    d.runs_total,
                    d.extras_type,
                    d.is_wicket,
                    d.wicket_kind,
                    out_id,
                    d.phase,
                    d.is_boundary,
                    d.is_six,
                    d.is_dot_ball,
                    d.is_legal_delivery,
                    d.cumulative_runs,
                    d.cumulative_wickets,
                ))

    print(f"  Inserting {len(all_rows):,} deliveries...")
    psycopg2.extras.execute_values(
        cursor,
        """INSERT INTO deliveries (
            innings_id, match_id, over_number, ball_in_over, legal_ball_number,
            batter_id, non_striker_id, bowler_id,
            runs_batter, runs_extras, runs_total, extras_type,
            is_wicket, wicket_kind, player_out_id,
            phase, is_boundary, is_six, is_dot_ball, is_legal_delivery,
            cumulative_runs, cumulative_wickets
        ) VALUES %s
        ON CONFLICT (innings_id, over_number, ball_in_over) DO NOTHING""",
        all_rows,
        page_size=2000,    # 2000 rows per SQL batch = ~148 SQL calls for 295K rows
    )
    delivery_count = cursor.rowcount
    print(f"  Deliveries inserted: ~{delivery_count:,} (approx — execute_values batched)")

    # SELECT wicket delivery IDs for dismissal insertion
    # (can't use RETURNING across multiple pages reliably)
    cursor.execute(
        "SELECT id, innings_id, over_number, ball_in_over FROM deliveries "
        "WHERE is_wicket = true AND innings_id = ANY(%s)",
        (list(innings_map.values()),)
    )
    wicket_delivery_map = {(row[1], row[2], row[3]): row[0] for row in cursor.fetchall()}
    return wicket_delivery_map


# ── Phase 6: Bulk-insert dismissals ───────────────────────────────────────────

def insert_all_dismissals(
    cursor,
    parsed_matches: list[ParsedMatch],
    match_db_ids: dict,
    innings_map: dict,
    wicket_delivery_map: dict,
    id_maps: dict,
) -> None:
    """Bulk-insert dismissal records for all wicket deliveries."""
    player_map = id_maps["player"]
    rows = []

    for p in parsed_matches:
        match_id = match_db_ids.get(p.cricsheet_file_id)
        if match_id is None:
            continue
        for inn in p.innings:
            innings_id = innings_map.get((match_id, inn.innings_number))
            if innings_id is None:
                continue
            for d in inn.deliveries:
                if not d.is_wicket:
                    continue
                key = (innings_id, d.over_number, d.ball_in_over)
                del_id = wicket_delivery_map.get(key)
                if del_id is None:
                    continue   # delivery was already in DB (ON CONFLICT)

                out_id    = player_map.get(d.player_out_id) if d.player_out_id else None
                bowler_id = player_map.get(d.bowler_id)
                fielder_ids = [player_map.get(fn) for fn in (d.fielder_names or [])[:2]]
                while len(fielder_ids) < 2:
                    fielder_ids.append(None)

                rows.append((
                    del_id,
                    out_id,
                    d.wicket_kind or "unknown",
                    bowler_id,
                    fielder_ids[0],
                    fielder_ids[1],
                ))

    if rows:
        psycopg2.extras.execute_values(
            cursor,
            """INSERT INTO dismissals (delivery_id, player_out_id, kind, bowler_id, fielder1_id, fielder2_id)
               VALUES %s ON CONFLICT DO NOTHING""",
            rows,
            page_size=500,
        )
    print(f"  Dismissals inserted: {len(rows)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    raw_dir = settings.data_raw_dir / "cricsheet" / args.competition
    if not raw_dir.exists():
        print(f"ERROR: No data at {raw_dir}. Run download_cricsheet.py first.")
        sys.exit(1)

    t0 = time.time()
    print(f"\nCricketIQ — Fast Bulk Ingest")
    print(f"Source: {raw_dir}")
    print(f"Target: {db_url.split('@')[-1]}")

    # Phase 1 — Parse
    parsed_matches, raw_jsons = parse_all_files(raw_dir, args.limit)
    if not parsed_matches:
        print("No matches to ingest.")
        return

    # Connect via psycopg2 directly (bypasses SQLAlchemy NullPool overhead)
    print(f"\nConnecting to database...")
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cursor = conn.cursor()
    cursor.execute("SELECT version()")
    print(f"  {cursor.fetchone()[0][:50]}")

    try:
        # Phase 2 — Reference tables
        print("\nPhase 2: Reference tables...")
        id_maps = load_reference_tables(cursor, parsed_matches)

        # Phase 3 — Matches
        print("\nPhase 3: Matches...")
        match_db_ids = insert_all_matches(cursor, parsed_matches, raw_jsons, id_maps)

        # Phase 4 — Innings
        print("\nPhase 4: Innings...")
        innings_map = insert_all_innings(cursor, parsed_matches, match_db_ids, id_maps)

        # Phase 5 — Deliveries
        print("\nPhase 5: Deliveries...")
        wicket_delivery_map = insert_all_deliveries(
            cursor, parsed_matches, match_db_ids, innings_map, id_maps
        )

        # Phase 6 — Dismissals
        print("\nPhase 6: Dismissals...")
        insert_all_dismissals(
            cursor, parsed_matches, match_db_ids, innings_map, wicket_delivery_map, id_maps
        )

        # Commit everything
        print("\nCommitting...")
        conn.commit()
        elapsed = time.time() - t0
        # Verify counts
        cursor.execute("SELECT COUNT(*) FROM matches")
        m_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM innings")
        i_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM deliveries")
        d_count = cursor.fetchone()[0]

        print(f"\n✅ Done in {elapsed:.0f}s ({elapsed/60:.1f} minutes)")
        print(f"   Matches   : {m_count:,}")
        print(f"   Innings   : {i_count:,}")
        print(f"   Deliveries: {d_count:,}")

    except Exception as e:
        conn.rollback()
        print(f"\n❌ Error — rolled back: {e}")
        logger.exception("Bulk ingest failed")
        sys.exit(1)
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
