"""
CricketIQ — Shared Test Fixtures
==================================
pytest loads this file automatically before any test runs.
Fixtures defined here are available to ALL test files without importing.

Usage in a test file:
    def test_something(sample_match_json):
        assert sample_match_json["info"]["teams"] == ["CSK", "GT"]
"""

import json
from pathlib import Path

import pytest

# Path to sample data committed in the repo
SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"


@pytest.fixture(scope="session")
def sample_match_json() -> dict:
    """
    Load the first available sample match JSON file.
    scope="session" means this fixture is created once per test session,
    not re-created for each test function — faster.
    """
    sample_files = list(SAMPLES_DIR.glob("*.json"))
    if not sample_files:
        pytest.skip(
            "No sample JSON files found in data/samples/. "
            "Run: python scripts/download_cricsheet.py --sample"
        )
    with open(sample_files[0], encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def all_sample_matches() -> list[dict]:
    """Load ALL sample match JSON files as a list of dicts."""
    sample_files = list(SAMPLES_DIR.glob("*.json"))
    if not sample_files:
        pytest.skip("No sample JSON files found in data/samples/.")
    matches = []
    for path in sample_files:
        with open(path, encoding="utf-8") as f:
            matches.append(json.load(f))
    return matches


@pytest.fixture
def minimal_delivery() -> dict:
    """
    A minimal valid delivery dict as it appears in Cricsheet JSON.
    Use this when testing parsers without needing a full match file.
    """
    return {
        "batter": "Virat Kohli",
        "bowler": "Jasprit Bumrah",
        "non_striker": "Rohit Sharma",
        "runs": {
            "batter": 4,
            "extras": 0,
            "total": 4,
        },
    }


@pytest.fixture
def delivery_with_wicket() -> dict:
    """A delivery that includes a wicket (caught dismissal)."""
    return {
        "batter": "Virat Kohli",
        "bowler": "Jasprit Bumrah",
        "non_striker": "Rohit Sharma",
        "runs": {
            "batter": 0,
            "extras": 0,
            "total": 0,
        },
        "wickets": [
            {
                "kind": "caught",
                "player_out": "Virat Kohli",
                "fielders": [{"name": "MS Dhoni"}],
            }
        ],
    }


@pytest.fixture
def delivery_with_wide() -> dict:
    """A wide delivery — should NOT count as a legal delivery."""
    return {
        "batter": "Virat Kohli",
        "bowler": "Jasprit Bumrah",
        "non_striker": "Rohit Sharma",
        "runs": {
            "batter": 0,
            "extras": 1,
            "total": 1,
        },
        "extras": {"wides": 1},
    }


@pytest.fixture
def mock_settings(monkeypatch):
    """
    Override settings for tests — prevents needing a real .env file in CI.
    Uses monkeypatch to set environment variables before Settings() loads.
    """
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/cricketiq_test")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("LLM_API_KEY", "fake-key-for-testing")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")   # Quiet during tests
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("N_SIMULATIONS", "100")   # Fast simulations in tests
