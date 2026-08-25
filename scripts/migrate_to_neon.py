"""
CricketIQ — Migrate Local PostgreSQL → Neon
============================================
This script dumps the local cricketiq database and restores it to Neon.

PREREQUISITES:
    1. pg_dump and psql must be in PATH (installed with PostgreSQL)
    2. LOCAL DATABASE_URL set in .env (your local Postgres)
    3. NEON_DATABASE_URL set in .env (your Neon connection string)
       Format: postgresql://user:password@ep-xxx.region.aws.neon.tech/cricketiq?sslmode=require

USAGE:
    python scripts/migrate_to_neon.py [--dry-run]

WHAT IT DOES:
    1. pg_dump local cricketiq → /tmp/cricketiq_dump.sql  (schema + data)
    2. Run alembic upgrade head against Neon (creates tables)
    3. psql against Neon with --data-only to restore rows

ALTERNATIVE (if pg_dump not available):
    Run the full ingestion pipeline directly against Neon:
        DATABASE_URL=<neon_url> python scripts/ingest_matches.py
        DATABASE_URL=<neon_url> python scripts/compute_distributions.py
        DATABASE_URL=<neon_url> python scripts/train_win_model.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description="Migrate local cricketiq DB to Neon")
    p.add_argument("--dry-run", action="store_true", help="Print commands only, don't execute")
    p.add_argument("--local-url",  default=os.getenv("DATABASE_URL"),      help="Local DB URL")
    p.add_argument("--neon-url",   default=os.getenv("NEON_DATABASE_URL"), help="Neon DB URL")
    return p.parse_args()


def _parse_pg_url(url: str) -> dict:
    """Parse a PostgreSQL connection URL into components."""
    parsed = urlparse(url.replace("postgres://", "postgresql://"))
    return {
        "host":     parsed.hostname or "localhost",
        "port":     str(parsed.port or 5432),
        "dbname":   parsed.path.lstrip("/").split("?")[0],
        "user":     parsed.username or "postgres",
        "password": parsed.password or "",
    }


def _run(cmd: list[str], env: dict | None = None, dry_run: bool = False) -> bool:
    """Run a shell command; returns True on success."""
    display_cmd = " ".join(cmd)
    if dry_run:
        print(f"[DRY RUN] {display_cmd}")
        return True

    print(f"$ {display_cmd}")
    full_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, env=full_env, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
    return result.returncode == 0


def step1_dump_local(local_url: str, dump_file: Path, dry_run: bool) -> bool:
    """pg_dump the local database to a SQL file."""
    print("\n── Step 1: Dumping local database ──")
    p = _parse_pg_url(local_url)
    env = {"PGPASSWORD": p["password"]}
    cmd = [
        "pg_dump",
        "--no-owner", "--no-privileges",
        "--format=plain",
        "--file", str(dump_file),
        "-h", p["host"], "-p", p["port"],
        "-U", p["user"], p["dbname"],
    ]
    return _run(cmd, env=env, dry_run=dry_run)


def step2_run_migrations(neon_url: str, dry_run: bool) -> bool:
    """Run Alembic migrations against Neon to create schema."""
    print("\n── Step 2: Running Alembic migrations on Neon ──")
    cmd = ["alembic", "upgrade", "head"]
    env = {"DATABASE_URL": neon_url}
    return _run(cmd, env=env, dry_run=dry_run)


def step3_restore_to_neon(neon_url: str, dump_file: Path, dry_run: bool) -> bool:
    """Restore data to Neon via psql."""
    print("\n── Step 3: Restoring data to Neon ──")
    p = _parse_pg_url(neon_url)
    env = {"PGPASSWORD": p["password"]}

    # psql needs the connection without ?sslmode=require in -d
    dbname = p["dbname"]
    sslmode = "require" if "neon.tech" in neon_url else "prefer"

    cmd = [
        "psql",
        "-h", p["host"], "-p", p["port"],
        "-U", p["user"], "-d", dbname,
        f"--set=sslmode={sslmode}",
        "-f", str(dump_file),
    ]
    return _run(cmd, env=env, dry_run=dry_run)


def main():
    args = parse_args()

    if not args.local_url:
        print("ERROR: --local-url or DATABASE_URL env var is required")
        sys.exit(1)
    if not args.neon_url:
        print("ERROR: --neon-url or NEON_DATABASE_URL env var is required")
        print("Set NEON_DATABASE_URL in .env:")
        print("  NEON_DATABASE_URL=postgresql://user:pass@ep-xxx.region.aws.neon.tech/cricketiq?sslmode=require")
        sys.exit(1)

    print("CricketIQ — Local → Neon Migration")
    print(f"  Source: {_masked(args.local_url)}")
    print(f"  Target: {_masked(args.neon_url)}")
    if args.dry_run:
        print("  Mode:   DRY RUN (no changes)")
    print()

    with tempfile.NamedTemporaryFile(suffix=".sql", delete=False, mode="w") as f:
        dump_file = Path(f.name)

    try:
        ok = step1_dump_local(args.local_url, dump_file, args.dry_run)
        if not ok:
            print("\nFailed at Step 1 — aborting")
            sys.exit(1)

        ok = step2_run_migrations(args.neon_url, args.dry_run)
        if not ok:
            print("\nFailed at Step 2 — aborting")
            sys.exit(1)

        ok = step3_restore_to_neon(args.neon_url, dump_file, args.dry_run)
        if not ok:
            print("\nFailed at Step 3 — aborting")
            sys.exit(1)

        print("\n✅ Migration complete!")
        print("Next steps:")
        print("  1. Update DATABASE_URL in .env to your Neon URL")
        print("  2. Run: streamlit run streamlit_app.py")
        print("  3. Verify the app loads data from Neon")

    finally:
        if dump_file.exists() and not args.dry_run:
            dump_file.unlink()
            print(f"Temp file {dump_file} removed.")


def _masked(url: str) -> str:
    if "@" in url:
        scheme_user, rest = url.split("@", 1)
        if ":" in scheme_user:
            scheme_user = scheme_user.rsplit(":", 1)[0] + ":***"
        return f"{scheme_user}@{rest}"
    return url


if __name__ == "__main__":
    main()
