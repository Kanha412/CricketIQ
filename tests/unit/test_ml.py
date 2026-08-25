"""
Tests for the ML and simulation layers.

ML tests: feature engineering logic (no DB needed).
Simulation tests: statistical properties of Monte Carlo output.
Model tests: only run if the model file exists (skipped in fresh envs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.ml.features import (
    build_inference_row,
    FEATURE_COLS,
    build_feature_matrix,
)


# ===========================================================================
# Feature engineering tests (no DB, no model needed)
# ===========================================================================

class TestBuildInferenceRow:
    """build_inference_row must produce correct features for known inputs."""

    def test_output_shape(self):
        row = build_inference_row(100, 3, 60, 50, "middle")
        assert row.shape == (1, 7)

    def test_correct_feature_count(self):
        row = build_inference_row(50, 2, 30, 80, "death")
        assert row.shape[1] == len(FEATURE_COLS)

    def test_phase_encoding_powerplay(self):
        row = build_inference_row(20, 0, 108, 160, "powerplay")
        phase_idx = FEATURE_COLS.index("phase_encoded")
        assert row[0, phase_idx] == 0  # powerplay = 0

    def test_phase_encoding_middle(self):
        row = build_inference_row(60, 2, 60, 120, "middle")
        phase_idx = FEATURE_COLS.index("phase_encoded")
        assert row[0, phase_idx] == 1  # middle = 1

    def test_phase_encoding_death(self):
        row = build_inference_row(150, 4, 12, 30, "death")
        phase_idx = FEATURE_COLS.index("phase_encoded")
        assert row[0, phase_idx] == 2  # death = 2

    def test_required_runs_clamps_to_zero(self):
        # Already won (required_runs negative) → should clamp to 0
        row = build_inference_row(200, 0, 6, -5, "death")
        req_idx = FEATURE_COLS.index("required_runs")
        assert row[0, req_idx] == 0.0

    def test_balls_remaining_clamps_to_zero(self):
        row = build_inference_row(100, 3, 0, 10, "death")
        br_idx = FEATURE_COLS.index("balls_remaining")
        assert row[0, br_idx] == 0.0

    def test_required_rr_caps_at_36(self):
        # Impossible chase (100 runs in 0 balls) → capped at 36
        row = build_inference_row(80, 5, 0, 100, "death")
        rrr_idx = FEATURE_COLS.index("required_rr")
        assert row[0, rrr_idx] == pytest.approx(36.0)

    def test_required_rr_sensible(self):
        # 30 runs in 12 balls = 5 overs needed, RRR = 30 / (12/6) = 15.0
        row = build_inference_row(120, 2, 12, 30, "death")
        rrr_idx = FEATURE_COLS.index("required_rr")
        assert row[0, rrr_idx] == pytest.approx(15.0)

    def test_dtype_float32(self):
        row = build_inference_row(50, 2, 36, 40, "middle")
        assert row.dtype == np.float32

    def test_unknown_phase_defaults_to_middle(self):
        row_middle  = build_inference_row(50, 2, 36, 40, "middle")
        row_unknown = build_inference_row(50, 2, 36, 40, "unknown")
        phase_idx   = FEATURE_COLS.index("phase_encoded")
        assert row_middle[0, phase_idx] == row_unknown[0, phase_idx] == 1


class TestBuildFeatureMatrix:
    def test_correct_shape(self):
        df = pd.DataFrame({col: [1.0, 2.0] for col in FEATURE_COLS})
        X = build_feature_matrix(df)
        assert X.shape == (2, len(FEATURE_COLS))

    def test_dtype(self):
        df = pd.DataFrame({col: [1.0] for col in FEATURE_COLS})
        X = build_feature_matrix(df)
        assert X.dtype == np.float32


# ===========================================================================
# Model inference tests (skipped if not yet trained)
# ===========================================================================

MODEL_PATH = Path("models/win_probability_model.joblib")

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model not yet trained")
class TestWinProbabilityModel:
    """These only run if models/win_probability_model.joblib exists."""

    @pytest.fixture(scope="class")
    def model(self):
        from src.ml.win_probability import WinProbabilityModel
        return WinProbabilityModel.load()

    def test_returns_float(self, model):
        p = model.predict(100, 3, 60, 50, "middle")
        assert isinstance(p, float)

    def test_probability_in_range(self, model):
        p = model.predict(100, 3, 60, 50, "middle")
        assert 0.0 <= p <= 1.0

    def test_certain_win_is_high(self, model):
        # 1 run needed, 6 balls, 0 wickets → very high probability
        p = model.predict(179, 0, 6, 1, "death")
        assert p > 0.5, f"Expected >0.5 for near-certain win, got {p}"

    def test_certain_loss_is_low(self, model):
        # 100 runs needed, 1 ball, 9 wickets → very low probability
        p = model.predict(80, 9, 1, 100, "death")
        assert p < 0.10, f"Expected <0.10 for near-certain loss, got {p}"

    def test_fresh_chase_near_fifty(self, model):
        # Start of a 160-target chase → neither side heavily favoured
        p = model.predict(0, 0, 120, 160, "powerplay")
        assert 0.3 < p < 0.7, f"Fresh chase should be ~50%, got {p}"


# ===========================================================================
# Monte Carlo simulation tests (no DB: test statistical properties only)
# ===========================================================================

DIST_PATH = Path("models/ball_distributions.joblib")

@pytest.mark.skipif(not DIST_PATH.exists(), reason="Distributions not yet computed")
class TestMonteCarloSimulator:
    @pytest.fixture(scope="class")
    def sim(self):
        from src.simulation.monte_carlo import MonteCarloSimulator
        return MonteCarloSimulator.load(seed=42)

    def test_full_innings_score_range(self, sim):
        result = sim.simulate(n_simulations=500)
        # IPL T20 scores realistically between 80 and 260
        assert result.mean_score > 100, f"Mean too low: {result.mean_score}"
        assert result.mean_score < 220, f"Mean too high: {result.mean_score}"

    def test_scores_non_negative(self, sim):
        result = sim.simulate(n_simulations=500)
        assert (result.scores >= 0).all()

    def test_scores_not_absurd(self, sim):
        result = sim.simulate(n_simulations=500)
        assert result.scores.max() < 400  # physically impossible T20 score

    def test_reproducible_with_same_seed(self, sim):
        from src.simulation.monte_carlo import MonteCarloSimulator
        sim2 = MonteCarloSimulator.load(seed=42)
        r1 = sim.simulate(n_simulations=100)
        r2 = sim2.simulate(n_simulations=100)
        np.testing.assert_array_equal(r1.scores, r2.scores)

    def test_partial_innings(self, sim):
        # 36 balls bowled, 58 runs, 2 wickets → simulate remaining 84 balls
        result = sim.simulate(
            n_simulations=500,
            balls_bowled=36,
            runs_so_far=58,
            wickets_fallen=2,
        )
        assert (result.scores >= 58).all()  # can't score fewer than already scored

    def test_probability_of_reaching(self, sim):
        result = sim.simulate(n_simulations=1000)
        # Probability of reaching 80 should be higher than reaching 200
        p80  = result.probability_of_reaching(80)
        p200 = result.probability_of_reaching(200)
        assert p80 > p200

    def test_percentile_ordering(self, sim):
        result = sim.simulate(n_simulations=500)
        assert result.percentile_10 <= result.median_score <= result.percentile_90

    def test_score_distribution_df_shape(self, sim):
        result = sim.simulate(n_simulations=500)
        df = result.score_distribution_df()
        assert not df.empty
        assert "score_range" in df.columns
        assert "probability" in df.columns
        assert abs(df["probability"].sum() - 1.0) < 0.01   # probs sum to ~1
