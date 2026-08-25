"""
CricketIQ — Ball Outcome Distributions
=========================================
Computes phase-specific probability distributions for ball outcomes
from the historical delivery data.

These distributions power the Monte Carlo simulation.

OUTCOMES MODELLED:
    Runs from bat: 0 (dot), 1, 2, 3, 4, 6
    Wicket: boolean, modelled independently

WHY INDEPENDENT WICKET MODEL:
    In reality, a wicket delivery often scores 0 runs, so runs and wickets
    are slightly correlated.  But modelling them independently is a good
    approximation and keeps the simulation simple enough to explain.

WHY PHASE-SPECIFIC:
    Ball outcomes differ dramatically by phase:
    - Powerplay: more 4s, fewer dots (field restrictions)
    - Death: more 6s, high dot %, very risky
    Averaging across phases would miss these patterns.

SAVED TO: models/ball_distributions.joblib
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.db.connection import get_connection
from src.logger import get_logger

logger = get_logger(__name__)

DISTRIBUTIONS_PATH = Path(__file__).parent.parent.parent / "models" / "ball_distributions.joblib"

# Outcomes we model (runs from bat)
RUN_OUTCOMES = [0, 1, 2, 3, 4, 6]


# ---------------------------------------------------------------------------
# SQL: pull legal delivery outcomes per phase
# ---------------------------------------------------------------------------
_DIST_QUERY = """
SELECT
    d.phase,
    d.runs_batter,
    d.is_wicket,
    COUNT(*) AS n_deliveries
FROM deliveries d
WHERE d.is_legal_delivery = TRUE
  AND d.is_wicket IS NOT NULL
GROUP BY d.phase, d.runs_batter, d.is_wicket
ORDER BY d.phase, d.runs_batter
"""


def compute_distributions() -> dict:
    """
    Query the DB and compute phase-specific ball outcome distributions.

    Returns a dict with structure:
    {
        'powerplay': {
            'run_probs':    np.array([p0, p1, p2, p3, p4, p6]),  # sums to 1
            'run_outcomes': [0, 1, 2, 3, 4, 6],
            'wicket_prob':  float,
            'n_deliveries': int,
        },
        'middle': { ... },
        'death':  { ... },
    }

    WHY A DICT OF ARRAYS not a DataFrame:
        The Monte Carlo simulation calls np.random.choice() millions of
        times per run.  Having the probabilities as numpy arrays avoids
        repeated conversion overhead in the inner loop.
    """
    logger.info("Computing ball outcome distributions from database...")
    with get_connection() as conn:
        raw = pd.read_sql(text(_DIST_QUERY), conn)

    if raw.empty:
        raise ValueError("No delivery data found — run ingestion first.")

    dist: dict = {}
    for phase in ["powerplay", "middle", "death"]:
        phase_df = raw[raw["phase"] == phase].copy()
        if phase_df.empty:
            logger.warning("No data for phase: %s", phase)
            continue

        total = phase_df["n_deliveries"].sum()

        # Run probabilities
        run_counts = []
        for outcome in RUN_OUTCOMES:
            n = phase_df[phase_df["runs_batter"] == outcome]["n_deliveries"].sum()
            run_counts.append(int(n))

        # Anything not in RUN_OUTCOMES (e.g. 5, rare) → bucket into nearest
        accounted = sum(run_counts)
        remainder = int(total) - accounted
        if remainder > 0:
            run_counts[1] += remainder  # add to singles (most common)

        run_probs = np.array(run_counts, dtype=np.float64)
        run_probs /= run_probs.sum()   # normalise to sum=1

        # Wicket probability (independent of runs)
        wicket_rows = phase_df[phase_df["is_wicket"] == True]["n_deliveries"].sum()
        wicket_prob = float(wicket_rows / total)

        dist[phase] = {
            "run_probs":    run_probs,
            "run_outcomes": RUN_OUTCOMES,
            "wicket_prob":  wicket_prob,
            "n_deliveries": int(total),
        }
        logger.info(
            "Phase %-10s | %6d deliveries | wicket_prob=%.3f | "
            "dot=%.1f%% | 4=%.1f%% | 6=%.1f%%",
            phase, total, wicket_prob,
            run_probs[0] * 100, run_probs[4] * 100, run_probs[5] * 100,
        )

    return dist


def save_distributions(dist: dict, path: Path = DISTRIBUTIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dist, path)
    logger.info("Ball distributions saved to %s", path)


def load_distributions(path: Path = DISTRIBUTIONS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Ball distributions not found at {path}.\n"
            "Compute them first:\n"
            "    python scripts/compute_distributions.py"
        )
    return joblib.load(path)


def is_computed(path: Path = DISTRIBUTIONS_PATH) -> bool:
    return path.exists()
