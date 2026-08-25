"""
CricketIQ — Monte Carlo Innings Simulation
============================================
Simulates N T20 innings using historical phase distributions.

ALGORITHM:
    For each simulation:
        1. Start: score=0, wickets=0, ball=0
        2. For each ball (1..120):
            a. Determine phase from ball number
            b. Sample run outcome from phase distribution
            c. Sample wicket independently from phase wicket probability
            d. If wicket: wickets += 1; if wickets == 10: stop
            e. score += sampled_runs
        3. Record final score

VECTORISATION:
    Instead of a Python loop per simulation, we pre-sample ALL runs and
    wicket events for ALL simulations at once:
        runs_matrix:    (N_SIMS, 120) int array
        wicket_matrix:  (N_SIMS, 120) bool array

    Then we use cumulative sums and argmax tricks to find the end of
    each innings without any Python loops.  10,000 sims run in < 0.5s.

    WHY THIS MATTERS: Streamlit re-runs on every widget interaction.
    A 10-second simulation would make the UI feel broken; <1s is invisible.

REPRODUCIBILITY:
    All randomness goes through np.random.default_rng(seed).
    Same seed = same results, which makes AI agent answers consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.simulation.distributions import load_distributions, RUN_OUTCOMES
from src.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class SimulationResult:
    """
    Full results from a Monte Carlo innings simulation.

    Attributes:
        scores:          1-D array of N final scores
        n_simulations:   How many sims were run
        mean_score:      Expected score
        median_score:    Median (less affected by outliers)
        percentile_10:   10th percentile (pessimistic outcome)
        percentile_90:   90th percentile (optimistic outcome)
        prob_above:      Dict: {target: probability of reaching target}
    """
    scores:         np.ndarray
    n_simulations:  int
    mean_score:     float
    median_score:   float
    percentile_10:  float
    percentile_90:  float
    prob_above:     dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_scores(cls, scores: np.ndarray) -> "SimulationResult":
        return cls(
            scores        = scores,
            n_simulations = len(scores),
            mean_score    = float(np.mean(scores)),
            median_score  = float(np.median(scores)),
            percentile_10 = float(np.percentile(scores, 10)),
            percentile_90 = float(np.percentile(scores, 90)),
        )

    def probability_of_reaching(self, target: int) -> float:
        """P(score >= target) from the simulation."""
        return float((self.scores >= target).mean())

    def score_distribution_df(self, bin_width: int = 5) -> pd.DataFrame:
        """
        Returns a DataFrame suitable for plotting a score histogram.
        Columns: score_range (str label), count, probability
        """
        bins = range(0, max(250, int(self.scores.max()) + bin_width + 1), bin_width)
        hist, edges = np.histogram(self.scores, bins=list(bins))
        labels = [f"{edges[i]}-{edges[i+1]-1}" for i in range(len(hist))]
        df = pd.DataFrame({
            "score_range":  labels,
            "count":        hist,
            "probability":  hist / self.n_simulations,
        })
        return df[df["count"] > 0].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------
class MonteCarloSimulator:
    """
    Vectorised T20 innings simulator.

    Usage:
        sim = MonteCarloSimulator.load()
        result = sim.simulate(n_simulations=10_000)
        print(f"Expected score: {result.mean_score:.0f}")

    Or with a partial innings (balls already bowled):
        result = sim.simulate(
            balls_bowled=36,          # powerplay done
            runs_so_far=58,
            wickets_fallen=2,
            n_simulations=10_000,
        )
    """

    T20_MAX_BALLS = 120
    T20_MAX_WICKETS = 10

    def __init__(self, distributions: dict, seed: int = 42) -> None:
        self.distributions = distributions
        self.seed          = seed
        self._build_phase_arrays()

    def _build_phase_arrays(self) -> None:
        """
        Pre-build per-ball numpy arrays for run probabilities and wicket probs.

        Ball 1–36   → powerplay distribution
        Ball 37–90  → middle distribution
        Ball 91–120 → death distribution

        WHY PER-BALL ARRAYS:
            Pre-computing these means the simulation inner loop just indexes
            into a pre-built array rather than doing a dict lookup per ball.
        """
        self._run_outcomes = np.array(RUN_OUTCOMES, dtype=np.int32)

        phases = self.distributions
        # For each ball (0-indexed), which phase?
        self._run_probs    = np.zeros((self.T20_MAX_BALLS, len(RUN_OUTCOMES)))
        self._wicket_probs = np.zeros(self.T20_MAX_BALLS)

        for ball_idx in range(self.T20_MAX_BALLS):
            over = ball_idx // 6 + 1          # 1-indexed over
            if over <= 6:
                phase = "powerplay"
            elif over <= 15:
                phase = "middle"
            else:
                phase = "death"

            if phase in phases:
                self._run_probs[ball_idx]    = phases[phase]["run_probs"]
                self._wicket_probs[ball_idx] = phases[phase]["wicket_prob"]
            else:
                # Fallback: uniform distribution
                self._run_probs[ball_idx]    = np.ones(len(RUN_OUTCOMES)) / len(RUN_OUTCOMES)
                self._wicket_probs[ball_idx] = 0.05

    @classmethod
    def load(cls, seed: int = 42) -> "MonteCarloSimulator":
        distributions = load_distributions()
        return cls(distributions, seed=seed)

    @classmethod
    def is_ready(cls) -> bool:
        from src.simulation.distributions import is_computed
        return is_computed()

    # ---------------------------------------------------------------------------
    # Core simulation
    # ---------------------------------------------------------------------------
    def simulate(
        self,
        n_simulations: int = 10_000,
        balls_bowled: int = 0,
        runs_so_far: int = 0,
        wickets_fallen: int = 0,
        target: int | None = None,
    ) -> SimulationResult:
        """
        Simulate the REMAINING balls of a T20 innings N times.

        Args:
            n_simulations:  Number of simulations to run
            balls_bowled:   Legal balls already bowled (0 = full innings simulation)
            runs_so_far:    Runs already scored (added to each simulation's total)
            wickets_fallen: Wickets already lost (affects when innings ends)
            target:         Optional chase target; used in prob_above calculation

        Returns:
            SimulationResult with score distribution

        WHY VECTORISED:
            Pre-sample entire (N, remaining_balls) matrices at once using numpy.
            Then use cumulative sums and masked operations to find end of each
            innings without any Python for-loops.
        """
        balls_remaining = self.T20_MAX_BALLS - balls_bowled
        wickets_available = self.T20_MAX_WICKETS - wickets_fallen

        if balls_remaining <= 0 or wickets_available <= 0:
            # Innings already complete
            scores = np.full(n_simulations, runs_so_far, dtype=np.int32)
            result = SimulationResult.from_scores(scores)
            if target is not None:
                result.prob_above[target] = result.probability_of_reaching(target)
            return result

        rng = np.random.default_rng(self.seed)

        # Slice per-ball arrays to only remaining balls
        start_ball = balls_bowled
        run_probs_slice    = self._run_probs[start_ball: start_ball + balls_remaining]     # (remaining, 6)
        wicket_probs_slice = self._wicket_probs[start_ball: start_ball + balls_remaining]  # (remaining,)

        # Sample runs for every sim and every ball at once: shape (N, remaining)
        run_matrix = self._sample_runs_vectorised(rng, n_simulations, balls_remaining, run_probs_slice)

        # Sample wickets for every sim and every ball: shape (N, remaining), bool
        wicket_matrix = rng.random((n_simulations, balls_remaining)) < wicket_probs_slice[np.newaxis, :]

        # Compute final scores accounting for all-out
        scores = self._compute_scores(
            run_matrix, wicket_matrix, wickets_available, runs_so_far
        )

        result = SimulationResult.from_scores(scores)
        if target is not None:
            result.prob_above[target] = result.probability_of_reaching(target)

        logger.debug(
            "Simulation complete: %d sims | mean=%.0f | p10=%.0f | p90=%.0f",
            n_simulations, result.mean_score, result.percentile_10, result.percentile_90,
        )
        return result

    def _sample_runs_vectorised(
        self,
        rng: np.random.Generator,
        n_sims: int,
        n_balls: int,
        run_probs_slice: np.ndarray,
    ) -> np.ndarray:
        """
        Sample run outcomes for all sims and all balls at once.

        For each ball position, the run probabilities differ (phase-specific).
        We loop over ball positions (max 120 iterations), but each step
        processes all N simulations at once — much faster than a per-sim loop.

        Returns:
            run_matrix: (n_sims, n_balls) int32 array of runs per delivery
        """
        run_matrix = np.empty((n_sims, n_balls), dtype=np.int32)
        outcomes   = np.array(RUN_OUTCOMES, dtype=np.int32)

        for ball_idx in range(n_balls):
            probs   = run_probs_slice[ball_idx]
            indices = rng.choice(len(outcomes), size=n_sims, p=probs)
            run_matrix[:, ball_idx] = outcomes[indices]

        return run_matrix

    def _compute_scores(
        self,
        run_matrix: np.ndarray,
        wicket_matrix: np.ndarray,
        wickets_available: int,
        runs_so_far: int,
    ) -> np.ndarray:
        """
        Given pre-sampled runs and wickets, compute the final score for
        each simulation, stopping when all wickets fall.

        Uses cumulative wicket sums to find the all-out ball for each sim,
        then masks out runs scored after all-out.

        Returns:
            scores: (n_sims,) int32 array of final scores
        """
        # Cumulative wickets per simulation per ball
        cum_wickets = np.cumsum(wicket_matrix, axis=1)  # (N, balls)

        # For each simulation, find the ball index where innings ends:
        # either all wickets fall or we run out of balls.
        # all-out ball = first ball where cum_wickets >= wickets_available
        all_out_mask = cum_wickets >= wickets_available   # (N, balls), bool

        # balls_valid[i, j] = True if ball j is still in play for sim i
        # i.e., innings hasn't ended before ball j
        # Shift the all_out_mask right by 1: once we're all out ON ball j,
        # that run still counts but ball j+1 onwards does not.
        innings_live = np.ones_like(all_out_mask, dtype=bool)
        innings_live[:, 1:] = ~all_out_mask[:, :-1]   # end AFTER the all-out ball

        # Apply the mask: zero out runs from dead balls
        masked_runs = run_matrix * innings_live.astype(np.int32)

        # Sum across balls to get per-sim runs in remaining innings
        remaining_runs = masked_runs.sum(axis=1)

        return (remaining_runs + runs_so_far).astype(np.int32)
