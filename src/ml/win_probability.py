"""
CricketIQ — Win Probability Inference
========================================
Loads the trained Logistic Regression model and provides a clean interface
for predicting win probability at any point in a T20 chase.

USAGE (after running scripts/train_win_model.py):

    from src.ml.win_probability import WinProbabilityModel
    model = WinProbabilityModel.load()
    prob = model.predict(
        runs_scored=85,
        wickets_fallen=3,
        balls_remaining=48,
        required_runs=65,
        phase='middle',
    )
    # prob = 0.42 → batting team has 42% chance of winning

WHY A CLASS (not a module-level function):
    The model is loaded from disk once and cached as an instance attribute.
    Module-level state is tricky with Streamlit's module reloading.  A class
    instance with @classmethod load() is explicit and testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

from src.ml.features import build_inference_row, FEATURE_COLS
from src.logger import get_logger

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

logger = get_logger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "win_probability_model.joblib"


class WinProbabilityModel:
    """
    Wrapper around the calibrated sklearn Pipeline for win probability.

    Attributes:
        pipeline: sklearn Pipeline (StandardScaler → CalibratedLogisticRegression)
        feature_cols: list of feature names (for debugging / explainability)
    """

    def __init__(self, pipeline) -> None:
        self.pipeline     = pipeline
        self.feature_cols = FEATURE_COLS

    # ---------------------------------------------------------------------------
    # Load / save
    # ---------------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "WinProbabilityModel":
        """
        Load the trained model from disk.

        Raises FileNotFoundError with a helpful message if not yet trained.
        WHY NOT SILENT DEFAULT: Silently returning 0.5 would hide the fact
        that training hasn't been run — better to fail loudly.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Win probability model not found at {path}.\n"
                "Train it first:\n"
                "    python scripts/train_win_model.py"
            )
        logger.info("Loading win probability model from %s", path)
        pipeline = joblib.load(path)
        return cls(pipeline)

    @classmethod
    def is_trained(cls, path: Path = MODEL_PATH) -> bool:
        """Return True if the model file exists on disk."""
        return path.exists()

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info("Win probability model saved to %s", path)

    # ---------------------------------------------------------------------------
    # Prediction interface
    # ---------------------------------------------------------------------------

    def predict(
        self,
        runs_scored: int,
        wickets_fallen: int,
        balls_remaining: int,
        required_runs: int,
        phase: str = "middle",
    ) -> float:
        """
        Predict the probability that the BATTING TEAM (chasing team) wins.

        Args:
            runs_scored:     Runs scored so far in the 2nd innings
            wickets_fallen:  Wickets lost so far
            balls_remaining: Legal deliveries remaining (T20 max = 120)
            required_runs:   Runs still needed to win
            phase:           Current phase: 'powerplay'|'middle'|'death'

        Returns:
            float in [0, 1]: probability the batting team wins.
            0.72 → "72% chance the chasing team wins"

        WHY RETURN THE CHASING TEAM'S PROBABILITY:
            The UI always shows "Team X's win probability" where Team X is
            batting.  Callers flip it for the fielding team: 1 - p.
        """
        X = build_inference_row(
            runs_scored=runs_scored,
            wickets_fallen=wickets_fallen,
            balls_remaining=balls_remaining,
            required_runs=required_runs,
            phase=phase,
        )
        # predict_proba returns [[P(lose), P(win)]] — index 1 is win
        proba = self.pipeline.predict_proba(X)[0][1]
        return float(round(proba, 4))

    def predict_over_trajectory(
        self,
        deliveries_df,
    ) -> list[dict]:
        """
        Given a second-innings deliveries DataFrame (all balls so far),
        compute win probability at the end of each completed over.

        Returns list of dicts:
            [{"over": 1, "runs": 12, "wickets": 0, "win_prob": 0.55}, ...]

        WHY TRAJECTORY:
            A single win probability is a snapshot.  A trajectory shows
            how momentum shifted — "MI were 80% through the powerplay then
            lost two quick wickets" is more insightful.
        """
        from src.ml.features import build_feature_matrix
        import pandas as pd

        rows = []
        for over in range(1, 21):
            over_df = deliveries_df[deliveries_df["over_number"] <= over]
            if over_df.empty:
                break

            legal = over_df[over_df["is_legal_delivery"]]
            if legal.empty:
                continue

            last_ball = legal.iloc[-1]
            runs      = int(last_ball.get("cumulative_runs", 0))
            wickets   = int(last_ball.get("cumulative_wickets", 0))
            ball_num  = int(last_ball.get("legal_ball_number", over * 6))
            target    = int(over_df["target_runs"].iloc[0]) if "target_runs" in over_df.columns else None

            if target is None:
                continue

            req      = max(0, target - runs)
            balls_rem = max(0, 120 - ball_num)
            phase_map = {r: "powerplay" if r <= 6 else ("death" if r > 15 else "middle")
                         for r in range(1, 21)}
            phase = phase_map[over]

            prob = self.predict(
                runs_scored=runs,
                wickets_fallen=wickets,
                balls_remaining=balls_rem,
                required_runs=req,
                phase=phase,
            )
            rows.append({
                "over":      over,
                "runs":      runs,
                "wickets":   wickets,
                "required":  req,
                "win_prob":  prob,
            })
        return rows
