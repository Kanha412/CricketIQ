"""
CricketIQ — Team Name Normalisation
=====================================
Same franchise, different names across seasons:
  Delhi Daredevils (2008-2018) → Delhi Capitals (2019+)
  Kings XI Punjab (2008-2020)  → Punjab Kings (2021+)
  Rising Pune Supergiant       → Rising Pune Supergiants  (typo)
  Royal Challengers Bangalore  → Royal Challengers Bengaluru (city renamed 2024)

NOT merged (genuinely different franchises):
  Gujarat Lions (2016-2017) vs Gujarat Titans (2022+)
  Deccan Chargers vs Sunrisers Hyderabad
  Pune Warriors vs Rising Pune Supergiant/s

Run once:
    uv run python scripts/normalize_teams.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

import psycopg2

DATABASE_URL = os.environ["DATABASE_URL"]

# (keep_id, canonical_name, [duplicate_ids_to_redirect])
MERGES = [
    (3,  "Delhi Capitals",              [4]),   # Delhi Daredevils → Delhi Capitals
    (13, "Punjab Kings",                [7]),   # Kings XI Punjab  → Punjab Kings
    (16, "Rising Pune Supergiants",     [15]),  # Rising Pune Supergiant → Supergiants
    (18, "Royal Challengers Bengaluru", [17]),  # RCB Bangalore → RCB Bengaluru
]

# Tables and columns that reference teams.id
TEAM_FK_COLUMNS = [
    ("matches", "team1_id"),
    ("matches", "team2_id"),
    ("matches", "winner_id"),
    ("matches", "toss_winner_id"),
    ("innings", "batting_team_id"),
    ("innings", "bowling_team_id"),
]


def run():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    total_rows_updated = 0
    total_teams_deleted = 0

    for keep_id, canonical_name, dup_ids in MERGES:
        if not dup_ids:
            continue

        rows_for_merge = 0

        # Update every FK column that points to a duplicate → point to canonical
        for table, col in TEAM_FK_COLUMNS:
            cur.execute(
                f"UPDATE {table} SET {col} = %s WHERE {col} = ANY(%s)",
                (keep_id, dup_ids),
            )
            rows_for_merge += cur.rowcount

        # Update player_match_stats if it exists
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'player_match_stats'
            )
        """)
        if cur.fetchone()[0]:
            cur.execute(
                "UPDATE player_match_stats SET team_id = %s WHERE team_id = ANY(%s)",
                (keep_id, dup_ids),
            )
            rows_for_merge += cur.rowcount

        total_rows_updated += rows_for_merge

        # Delete the duplicate team rows
        cur.execute("DELETE FROM teams WHERE id = ANY(%s)", (dup_ids,))
        n_deleted = cur.rowcount
        total_teams_deleted += n_deleted

        print(f"  OK '{canonical_name}': "
              f"{rows_for_merge} row(s) redirected, {n_deleted} duplicate(s) removed")

    conn.commit()
    cur.close()
    conn.close()

    print(f"\nDone — {total_rows_updated} rows updated, "
          f"{total_teams_deleted} duplicate team rows deleted.")

    # Verify final team list
    conn2 = psycopg2.connect(DATABASE_URL)
    cur2 = conn2.cursor()
    cur2.execute("SELECT id, name FROM teams ORDER BY name")
    print("\nFinal team list:")
    for row in cur2.fetchall():
        print(f"  {row[0]:>3}  {row[1]}")
    cur2.close()
    conn2.close()


if __name__ == "__main__":
    print("Normalising team names in Neon DB...\n")
    run()
