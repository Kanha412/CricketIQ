"""
CricketIQ — Venue Name Normalisation
======================================
Cricsheet records venue names inconsistently across seasons:
  "M Chinnaswamy Stadium" / "M Chinnaswamy Stadium, Bengaluru" / "M.Chinnaswamy Stadium"
  all refer to the same ground.

This script:
  1. Defines a merge map: (canonical_id, canonical_name, city) → [old_ids_to_redirect]
  2. Updates matches.venue_id to point to the canonical venue
  3. Deletes the now-unused duplicate venue rows

Run once after ingestion:
    uv run python scripts/normalize_venues.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

import psycopg2

DATABASE_URL = os.environ["DATABASE_URL"]

# ---------------------------------------------------------------------------
# Merge map
# Each entry: (keep_id, clean_name, clean_city, [duplicate_ids_to_redirect])
#
# RULE: keep_id is the canonical row we keep.
#       All matches pointing to a duplicate_id get redirected to keep_id.
#       The duplicate venue rows are then deleted.
# ---------------------------------------------------------------------------
MERGES = [
    # M Chinnaswamy Stadium, Bengaluru
    (8,  "M Chinnaswamy Stadium",       "Bengaluru",
         [5, 33]),

    # Arun Jaitley Stadium, Delhi  (formerly Feroz Shah Kotla)
    (16, "Arun Jaitley Stadium",         "Delhi",
         [24, 9]),    # 9 = "Feroz Shah Kotla"

    # Brabourne Stadium, Mumbai
    (52, "Brabourne Stadium",            "Mumbai",
         [26]),

    # Dr DY Patil Sports Academy, Navi Mumbai
    (43, "Dr DY Patil Sports Academy",   "Navi Mumbai",
         [27]),

    # ACA-VDCA Cricket Stadium, Visakhapatnam
    (17, "ACA-VDCA Cricket Stadium",     "Visakhapatnam",
         [38]),

    # Eden Gardens, Kolkata
    (7,  "Eden Gardens",                 "Kolkata",
         [29]),

    # Himachal Pradesh Cricket Association Stadium, Dharamsala
    (56, "HPCA Stadium",                 "Dharamsala",
         [36]),

    # MA Chidambaram Stadium (Chepauk), Chennai
    (14, "MA Chidambaram Stadium",       "Chennai",
         [42, 21]),

    # Maharaja Yadavindra Singh International Cricket Stadium (New Chandigarh / Mullanpur)
    (37, "Maharaja Yadavindra Singh Stadium", "New Chandigarh",
         [39]),

    # Maharashtra Cricket Association Stadium, Pune
    (2,  "Maharashtra Cricket Association Stadium", "Pune",
         [28]),

    # Narendra Modi Stadium, Ahmedabad  (formerly Sardar Patel / Motera)
    (23, "Narendra Modi Stadium",        "Ahmedabad",
         [53]),

    # Punjab Cricket Association IS Bindra Stadium, Mohali
    (12, "Punjab Cricket Association Stadium", "Mohali",
         [10, 30, 41]),

    # Rajiv Gandhi International Stadium, Hyderabad
    (13, "Rajiv Gandhi International Stadium", "Hyderabad",
         [1, 32]),

    # Sawai Mansingh Stadium, Jaipur
    (15, "Sawai Mansingh Stadium",       "Jaipur",
         [35]),

    # Shaheed Veer Narayan Singh International Stadium, Raipur
    (59, "Shaheed Veer Narayan Singh International Stadium", "Raipur",
         [40]),

    # Wankhede Stadium, Mumbai
    (6,  "Wankhede Stadium",             "Mumbai",
         [22]),

    # Zayed / Sheikh Zayed Cricket Stadium, Abu Dhabi
    (18, "Zayed Cricket Stadium",        "Abu Dhabi",
         [25]),
]


def run():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    total_matches_updated = 0
    total_venues_deleted  = 0

    for keep_id, clean_name, clean_city, dup_ids in MERGES:
        if not dup_ids:
            continue

        # 1. Redirect matches from duplicate venue_ids → canonical
        cur.execute(
            "UPDATE matches SET venue_id = %s WHERE venue_id = ANY(%s)",
            (keep_id, dup_ids),
        )
        n_updated = cur.rowcount
        total_matches_updated += n_updated

        # 2. Update the canonical venue row to use the clean name/city
        cur.execute(
            "UPDATE venues SET name = %s, city = %s WHERE id = %s",
            (clean_name, clean_city, keep_id),
        )

        # 3. Delete duplicate venue rows
        cur.execute("DELETE FROM venues WHERE id = ANY(%s)", (dup_ids,))
        n_deleted = cur.rowcount
        total_venues_deleted += n_deleted

        print(f"  OK '{clean_name}' ({clean_city}): "
              f"{n_updated} match(es) redirected, {n_deleted} duplicate(s) removed")

    conn.commit()
    cur.close()
    conn.close()

    print(f"\nDone — {total_matches_updated} match rows updated, "
          f"{total_venues_deleted} duplicate venue rows deleted.")

    # Verify
    conn2 = psycopg2.connect(DATABASE_URL)
    cur2  = conn2.cursor()
    cur2.execute("SELECT COUNT(*) FROM venues")
    print(f"Venues remaining: {cur2.fetchone()[0]}")
    cur2.close()
    conn2.close()


if __name__ == "__main__":
    print("Normalising venue names in Neon DB...\n")
    run()
