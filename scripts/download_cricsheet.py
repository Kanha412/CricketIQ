"""
CricketIQ — Cricsheet Data Downloader
======================================
Downloads ball-by-ball cricket match data from cricsheet.org.

Usage:
    python scripts/download_cricsheet.py --competitions ipl
    python scripts/download_cricsheet.py --competitions ipl bbl t20s
    python scripts/download_cricsheet.py --competitions ipl --incremental
    python scripts/download_cricsheet.py --list-competitions

How it works:
    1. Downloads a ZIP file for each requested competition from cricsheet.org
    2. Extracts individual match JSON files to data/raw/cricsheet/{competition}/
    3. Tracks processed files in data/raw/manifest.json (skips on re-run)
    4. --incremental flag skips competitions already downloaded today
"""

import json
import sys
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

# ── Add project root to path so we can import src/ ───────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.logger import get_logger

logger = get_logger("scripts.download_cricsheet")
console = Console()

# ── Competition → Download URL mapping ───────────────────────────────────────
# All URLs are from cricsheet.org/downloads/ — no scraping, direct links.
COMPETITION_URLS: dict[str, str] = {
    "ipl":   "https://cricsheet.org/downloads/ipl_json.zip",
    "t20s":  "https://cricsheet.org/downloads/t20s_json.zip",    # T20 Internationals
    "odis":  "https://cricsheet.org/downloads/odis_json.zip",    # ODI Internationals
    "tests": "https://cricsheet.org/downloads/tests_json.zip",   # Test matches
    "bbl":   "https://cricsheet.org/downloads/bbl_json.zip",     # Big Bash League
    "psl":   "https://cricsheet.org/downloads/psl_json.zip",     # Pakistan Super League
    "wpl":   "https://cricsheet.org/downloads/wpl_json.zip",     # Women's Premier League
    "sa20":  "https://cricsheet.org/downloads/sa20_json.zip",    # SA20
    "cpl":   "https://cricsheet.org/downloads/cpl_json.zip",     # Caribbean Premier League
    "lpl":   "https://cricsheet.org/downloads/lpl_json.zip",     # Lanka Premier League
}

MANIFEST_PATH = settings.data_raw_dir / "manifest.json"


# ── Manifest helpers ──────────────────────────────────────────────────────────

def load_manifest() -> dict:
    """
    Load the download manifest.
    The manifest tracks which files have been downloaded per competition.

    Structure:
        {
          "ipl": {
            "last_download": "2024-01-15",
            "file_count": 1034,
            "files": ["1234567", "1234568", ...]   ← match IDs (no .json)
          }
        }
    """
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    """Save the manifest to disk."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def get_known_files(manifest: dict, competition: str) -> set[str]:
    """Return set of already-downloaded match IDs for a competition."""
    return set(manifest.get(competition, {}).get("files", []))


# ── Download helpers ──────────────────────────────────────────────────────────

def download_zip(url: str, competition: str) -> bytes | None:
    """
    Download a ZIP file from the given URL.
    Streams the download to handle large files (20-100MB) without filling RAM.

    Returns:
        ZIP file contents as bytes, or None if download failed.
    """
    logger.info("Downloading %s from %s", competition, url)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Downloading {competition}..."),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
                if response.status_code == 404:
                    console.print(f"[red]✗[/red] {competition}: URL not found (404). Check competition name.")
                    return None
                if response.status_code != 200:
                    console.print(f"[red]✗[/red] {competition}: HTTP {response.status_code}")
                    return None

                total = int(response.headers.get("content-length", 0))
                task = progress.add_task("download", total=total if total > 0 else None)

                buffer = BytesIO()
                for chunk in response.iter_bytes(chunk_size=65536):  # 64KB chunks
                    buffer.write(chunk)
                    progress.update(task, advance=len(chunk))

        return buffer.getvalue()

    except httpx.ConnectError:
        console.print(f"[red]✗[/red] Could not connect to cricsheet.org. Check internet connection.")
        logger.error("Connection error downloading %s", competition)
        return None
    except httpx.TimeoutException:
        console.print(f"[red]✗[/red] Download timed out for {competition}.")
        logger.error("Timeout downloading %s", competition)
        return None


def extract_zip(
    zip_bytes: bytes,
    competition: str,
    known_files: set[str],
    incremental: bool,
) -> tuple[list[str], int]:
    """
    Extract JSON match files from a ZIP archive.

    Args:
        zip_bytes:    Raw bytes of the downloaded ZIP file.
        competition:  Competition name (used to determine output directory).
        known_files:  Set of match IDs already on disk (skip these if incremental).
        incremental:  If True, skip files already in known_files.

    Returns:
        Tuple of (list of newly extracted match IDs, count of skipped files).
    """
    output_dir = settings.data_raw_dir / "cricsheet" / competition
    output_dir.mkdir(parents=True, exist_ok=True)

    newly_extracted: list[str] = []
    skipped = 0

    try:
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            # Get all .json files (skip README.txt, etc.)
            json_files = [name for name in zf.namelist() if name.endswith(".json")]

            logger.info("ZIP contains %d match files for %s", len(json_files), competition)

            with Progress(
                SpinnerColumn(),
                TextColumn(f"[bold green]Extracting {competition}..."),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total} files)"),
                console=console,
            ) as progress:
                task = progress.add_task("extract", total=len(json_files))

                for filename in json_files:
                    match_id = filename.replace(".json", "")

                    # Skip files we've already extracted (incremental mode)
                    if incremental and match_id in known_files:
                        skipped += 1
                        progress.advance(task)
                        continue

                    # Extract the file
                    file_bytes = zf.read(filename)
                    output_path = output_dir / filename
                    output_path.write_bytes(file_bytes)
                    newly_extracted.append(match_id)
                    progress.advance(task)

    except zipfile.BadZipFile:
        console.print(f"[red]✗[/red] Downloaded file for {competition} is not a valid ZIP.")
        logger.error("Bad ZIP file for %s", competition)
        return [], 0

    return newly_extracted, skipped


# ── Main download flow ────────────────────────────────────────────────────────

def download_competition(
    competition: str,
    manifest: dict,
    incremental: bool,
) -> bool:
    """
    Download and extract data for a single competition.

    Returns:
        True if successful, False on failure.
    """
    url = COMPETITION_URLS.get(competition)
    if not url:
        console.print(f"[yellow]![/yellow] Unknown competition '{competition}'. Use --list-competitions to see options.")
        return False

    # Check if already downloaded today (incremental mode)
    comp_manifest = manifest.get(competition, {})
    last_download = comp_manifest.get("last_download", "")
    today = str(date.today())

    if incremental and last_download == today:
        console.print(f"[yellow]→[/yellow] {competition}: Already downloaded today. Use --no-incremental to force re-download.")
        return True

    # Download
    zip_bytes = download_zip(url, competition)
    if zip_bytes is None:
        return False

    # Extract
    known_files = get_known_files(manifest, competition)
    newly_extracted, skipped = extract_zip(zip_bytes, competition, known_files, incremental)

    # Update manifest
    all_files = list(known_files | set(newly_extracted))
    manifest[competition] = {
        "last_download": today,
        "file_count": len(all_files),
        "files": sorted(all_files),
    }
    save_manifest(manifest)

    # Report
    output_dir = settings.data_raw_dir / "cricsheet" / competition
    console.print(
        f"[green]✓[/green] {competition}: "
        f"[bold]{len(newly_extracted)}[/bold] new files extracted, "
        f"{skipped} skipped, "
        f"total {manifest[competition]['file_count']} matches in [dim]{output_dir}[/dim]"
    )
    logger.info(
        "Completed %s: %d new, %d skipped, %d total",
        competition, len(newly_extracted), skipped, manifest[competition]["file_count"]
    )
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--competitions", "-c",
    multiple=True,
    default=["ipl"],
    help="Which competitions to download. Can specify multiple: --competitions ipl bbl",
    show_default=True,
)
@click.option(
    "--incremental/--no-incremental",
    default=True,
    help="Skip files already downloaded. Use --no-incremental to re-download everything.",
    show_default=True,
)
@click.option(
    "--list-competitions",
    is_flag=True,
    default=False,
    help="List all available competitions and exit.",
)
def main(
    competitions: tuple[str, ...],
    incremental: bool,
    list_competitions: bool,
) -> None:
    """
    Download Cricsheet ball-by-ball match data.

    Data is saved to data/raw/cricsheet/{competition}/ as individual JSON files.
    A manifest at data/raw/manifest.json tracks what has been downloaded.

    Examples:

    \b
        # Download IPL data (default)
        python scripts/download_cricsheet.py

    \b
        # Download multiple competitions
        python scripts/download_cricsheet.py --competitions ipl bbl t20s

    \b
        # Force re-download (ignore manifest)
        python scripts/download_cricsheet.py --competitions ipl --no-incremental
    """
    # Show available competitions
    if list_competitions:
        table = Table(title="Available Cricsheet Competitions", show_header=True)
        table.add_column("Key", style="bold cyan")
        table.add_column("Competition")
        table.add_column("URL")
        for key, url in COMPETITION_URLS.items():
            table.add_row(key, key.upper(), url)
        console.print(table)
        console.print("\nUsage: python scripts/download_cricsheet.py --competitions ipl bbl")
        return

    console.print(f"\n[bold]CricketIQ — Cricsheet Data Downloader[/bold]")
    console.print(f"Competitions : [cyan]{', '.join(competitions)}[/cyan]")
    console.print(f"Mode         : [cyan]{'incremental (skip existing)' if incremental else 'full re-download'}[/cyan]")
    console.print(f"Output dir   : [dim]{settings.data_raw_dir}[/dim]\n")

    manifest = load_manifest()
    success_count = 0

    for competition in competitions:
        comp = competition.lower().strip()
        success = download_competition(comp, manifest, incremental)
        if success:
            success_count += 1

    console.print(f"\n[bold]Done.[/bold] {success_count}/{len(competitions)} competitions downloaded successfully.")

    if success_count > 0:
        console.print(
            "\n[dim]Next step: Run the ingestion pipeline to load this data into PostgreSQL:[/dim]"
        )
        console.print("[bold]  python scripts/ingest_matches.py[/bold]\n")


if __name__ == "__main__":
    main()
