"""
CricketIQ — Core Cricket Metric Formulas
==========================================
All primitive cricket calculations live here as plain functions.
They take scalar numbers and return a float (or None when undefined).

WHY SCALARS, NOT DATAFRAMES:
    Keeping formulas at the scalar level means they are trivially testable
    (no DataFrame fixtures needed), reusable across batting.py / bowling.py /
    matchup.py, and easy to reason about.  The higher-level modules call
    these inside .apply() or direct arithmetic on Series.

CONVENTION: return None when the denominator is 0 or the result is
    mathematically undefined — callers decide how to display it (e.g. "—").
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Batting metrics
# ---------------------------------------------------------------------------

def batting_average(runs: int | float, dismissals: int) -> float | None:
    """
    Runs scored ÷ times dismissed.
    Returns None if never dismissed (avoids divide-by-zero; display as '∞' or 'DNO').

    WHY: Average is the most stable long-term batting quality metric because
    it normalises for how many times you got out.  Strike rate alone is noisy
    in small samples.
    """
    if dismissals == 0:
        return None
    return round(runs / dismissals, 2)


def strike_rate(runs: int | float, balls_faced: int) -> float | None:
    """
    (Runs ÷ Balls faced) × 100.
    Returns None for 0 balls (avoids divide-by-zero).

    WHY: In T20s this is arguably more important than average — a 30-ball 40
    beats a 50-ball 40 every time.
    """
    if balls_faced == 0:
        return None
    return round((runs / balls_faced) * 100, 2)


def boundary_percentage(boundaries: int, balls_faced: int) -> float | None:
    """
    Percentage of balls that went for a boundary (4 or 6).
    High boundary % indicates a player attacks the ball rather than rotating strike.
    """
    if balls_faced == 0:
        return None
    return round((boundaries / balls_faced) * 100, 2)


def dot_ball_percentage(dot_balls: int, balls_faced: int) -> float | None:
    """
    Percentage of balls faced that produced no run.
    Low dot-ball % for a batter means they keep the scoreboard moving.
    """
    if balls_faced == 0:
        return None
    return round((dot_balls / balls_faced) * 100, 2)


# ---------------------------------------------------------------------------
# Bowling metrics
# ---------------------------------------------------------------------------

def economy_rate(runs_conceded: int | float, legal_deliveries: int) -> float | None:
    """
    Runs conceded per 6 legal deliveries (one over).
    Returns None for 0 deliveries.

    WHY: Economy is the primary T20 bowling metric because restricting runs
    in 20 overs is usually more impactful than taking wickets in a high-chase.
    """
    if legal_deliveries == 0:
        return None
    overs = legal_deliveries / 6
    return round(runs_conceded / overs, 2)


def bowling_average(runs_conceded: int | float, wickets: int) -> float | None:
    """
    Runs conceded ÷ wickets taken.
    Returns None for 0 wickets (a wicket has infinite cost if you never take one).
    """
    if wickets == 0:
        return None
    return round(runs_conceded / wickets, 2)


def bowling_strike_rate(legal_deliveries: int, wickets: int) -> float | None:
    """
    Legal deliveries bowled ÷ wickets taken.
    Balls needed per wicket — lower is better.
    Returns None for 0 wickets.
    """
    if wickets == 0:
        return None
    return round(legal_deliveries / wickets, 2)


def bowling_dot_percentage(dot_balls: int, legal_deliveries: int) -> float | None:
    """
    Percentage of legal deliveries that were dot balls.
    High dot % creates pressure and forces risky shots.
    """
    if legal_deliveries == 0:
        return None
    return round((dot_balls / legal_deliveries) * 100, 2)


# ---------------------------------------------------------------------------
# Match/innings-level metrics
# ---------------------------------------------------------------------------

def run_rate(runs: int | float, overs_completed: float) -> float | None:
    """
    Runs per over at a given point in an innings.
    Used for required-run-rate calculations and momentum tracking.
    """
    if overs_completed == 0:
        return None
    return round(runs / overs_completed, 2)


def required_run_rate(target: int, runs_scored: int, overs_remaining: float) -> float | None:
    """
    Runs still needed ÷ overs remaining.
    Returns None when no overs remain (game is over).
    """
    if overs_remaining <= 0:
        return None
    return round((target - runs_scored) / overs_remaining, 2)


def net_run_rate(
    runs_scored: int, overs_faced: float,
    runs_conceded: int, overs_bowled: float,
) -> float | None:
    """
    NRR = (runs scored / overs faced) – (runs conceded / overs bowled).
    Used for tournament standings tiebreakers.
    Returns None if either overs value is 0.
    """
    if overs_faced == 0 or overs_bowled == 0:
        return None
    return round((runs_scored / overs_faced) - (runs_conceded / overs_bowled), 3)


# ---------------------------------------------------------------------------
# Phase & context helpers
# ---------------------------------------------------------------------------

PHASE_LABELS: dict[str, str] = {
    "powerplay": "Powerplay (1–6)",
    "middle":    "Middle (7–15)",
    "death":     "Death (16–20)",
}


def phase_label(phase: str) -> str:
    """Human-readable label for a phase code."""
    return PHASE_LABELS.get(phase, phase.title())


def overs_from_deliveries(legal_deliveries: int) -> float:
    """
    Convert a count of legal deliveries to overs (e.g. 7 balls → 1.1 overs).
    Used for display and economy calculations.
    """
    complete = legal_deliveries // 6
    remainder = legal_deliveries % 6
    return complete + remainder / 10  # cricket notation: 1.3 means 1 over 3 balls
