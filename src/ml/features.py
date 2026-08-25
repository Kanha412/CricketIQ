"""
CricketIQ — Win Probability Feature Engineering
=================================================
Builds the training DataFrame from the deliveries + innings + matches tables.

WHY SQL-FIRST FEATURE ENGINEERING:
    Doing the aggregations in SQL (cumulative runs, required runs, RR) is ~10x
    faster than Python loops over 295K rows.  The query returns one row per
    legal 2nd-innings delivery, already feature-ready.

FEATURE SET (7 features):
    runs_scored      — cumulative runs at this ball
    wickets_fallen   — cumulative wickets lost
    balls_remaining  — legal balls left in innings (T20: 120 - ball_number)
    required_runs    — runs still needed
    required_rr      — required run rate (runs needed ÷ overs remaining)
    current_rr       — run rate so far (runs scored ÷ overs faced)
    phase_encoded    — 0=powerplay, 1=middle, 2=death

TARGET:
    label = 1 if batting team won, 0 otherwise

SCOPE (why only 2nd innings):
    The interesting win probability question is "can the chasing team get
    there?" — a known target makes the features meaningful.  1st-innings
    win probability is less actionable (there's no target yet).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sqlalchemy import text

from src.db.connection import get_connection
from src.logger import get_logger

logger = get_logger(__name__)

FEATURE_COLS = [
    "runs_scored",
    "wickets_fallen",
    "balls_remaining",
    "required_runs",
    "required_rr",
    "current_rr",
    "phase_encoded",
]
TARGET_COL = "label"


# ---------------------------------------------------------------------------
# SQL query — returns one row per legal 2nd-innings delivery
# ---------------------------------------------------------------------------
_TRAINING_QUERY = """
SELECT
    d.match_id,
    d.innings_id,
    d.legal_ball_number                                             AS ball_number,
    d.cumulative_runs                                               AS runs_scored,
    d.cumulative_wickets                                            AS wickets_fallen,

    -- balls remaining: cap at 0 so it never goes negative
    GREATEST(0, 120 - d.legal_ball_number)                          AS balls_remaining,

    -- required runs: cap at 0 (already won if negative)
    GREATEST(0, i.target_runs - d.cumulative_runs)                  AS required_runs,

    -- required run rate (runs per over): infinity if no balls left → cap at 36 (6 runs × 6 overs)
    CASE
        WHEN (120 - d.legal_ball_number) > 0
        THEN LEAST(36.0,
             GREATEST(0, i.target_runs - d.cumulative_runs)::float
             / ((120.0 - d.legal_ball_number) / 6.0))
        ELSE 36.0
    END                                                             AS required_rr,

    -- current run rate: 0 if no balls yet
    CASE
        WHEN d.legal_ball_number > 0
        THEN d.cumulative_runs::float / (d.legal_ball_number::float / 6.0)
        ELSE 0.0
    END                                                             AS current_rr,

    -- phase encoding: powerplay=0, middle=1, death=2
    CASE d.phase
        WHEN 'powerplay' THEN 0
        WHEN 'middle'    THEN 1
        WHEN 'death'     THEN 2
        ELSE 1
    END                                                             AS phase_encoded,

    -- label: 1 = batting team wins
    CASE WHEN i.batting_team_id = m.winner_id THEN 1 ELSE 0 END    AS label

FROM deliveries d
JOIN innings  i ON i.id  = d.innings_id
JOIN matches  m ON m.id  = d.match_id
WHERE
    i.innings_number = 2          -- 2nd innings (chasing) only
    AND i.target_runs IS NOT NULL -- must have a target
    AND d.is_legal_delivery = TRUE
    AND i.is_super_over   = FALSE  -- exclude super-over chases
    AND m.winner_id IS NOT NULL   -- completed matches only
ORDER BY d.match_id, d.legal_ball_number
"""


def load_training_data() -> pd.DataFrame:
    """
    Pull all 2nd-innings ball-by-ball data from the DB and return a
    feature-ready DataFrame.

    Returns:
        DataFrame with columns: match_id, innings_id, ball_number,
                                 runs_scored, wickets_fallen, balls_remaining,
                                 required_runs, required_rr, current_rr,
                                 phase_encoded, label

    WHY WE DON'T SAMPLE: We have ~120K 2nd-innings deliveries — small enough
    to fit in memory, large enough for LR to converge well.  No sampling needed.
    """
    logger.info("Loading win probability training data from database...")
    with get_connection() as conn:
        df = pd.read_sql(text(_TRAINING_QUERY), conn)

    logger.info("Loaded %d rows for win probability training", len(df))
    _log_class_balance(df)
    return df


def _log_class_balance(df: pd.DataFrame) -> None:
    if "label" not in df.columns or df.empty:
        return
    pos = df["label"].sum()
    total = len(df)
    logger.info(
        "Class balance: %d batting-team wins (%.1f%%), %d losses (%.1f%%)",
        pos, pos / total * 100, total - pos, (total - pos) / total * 100,
    )


# ---------------------------------------------------------------------------
# Feature matrix builder (used at inference too)
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Extract the feature matrix X from a training (or inference) DataFrame.
    Column order must match FEATURE_COLS exactly.
    """
    return df[FEATURE_COLS].to_numpy(dtype=np.float32)


def build_inference_row(
    runs_scored: int,
    wickets_fallen: int,
    balls_remaining: int,
    required_runs: int,
    phase: str,
) -> np.ndarray:
    """
    Build a single feature vector for real-time inference.

    Args:
        runs_scored:     Cumulative runs in the current innings
        wickets_fallen:  Wickets lost so far
        balls_remaining: Legal deliveries left (max 120 for T20)
        required_runs:   Runs still needed to win
        phase:           'powerplay' | 'middle' | 'death'

    Returns:
        shape (1, 7) numpy array ready for model.predict_proba()

    WHY A SEPARATE INFERENCE FUNCTION:
        Inference uses scalar inputs (from the AI agent or Streamlit form),
        not a DataFrame.  This function applies the same transformations as
        the SQL query so inference is consistent with training.
    """
    runs_needed = max(0, required_runs)
    balls_left  = max(0, balls_remaining)
    overs_left  = balls_left / 6.0

    required_rr = (
        min(36.0, runs_needed / overs_left) if overs_left > 0 else 36.0
    )
    balls_used  = max(1, 120 - balls_left)
    current_rr  = runs_scored / (balls_used / 6.0)

    phase_map   = {"powerplay": 0, "middle": 1, "death": 2}
    phase_enc   = phase_map.get(phase, 1)

    row = np.array([[
        runs_scored,
        wickets_fallen,
        balls_left,
        runs_needed,
        required_rr,
        current_rr,
        phase_enc,
    ]], dtype=np.float32)
    return row
