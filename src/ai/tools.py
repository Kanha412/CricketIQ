"""
CricketIQ — Agent Tool Functions
===================================
Each function here is a "tool" the AI agent can call.

CONTRACT:
    - Input:  simple Python scalars (str, int, float) — what an LLM can pass as JSON args
    - Output: JSON-serialisable dict — what the LLM reads and narrates
    - No DataFrames, no matplotlib Figures, no numpy arrays in return values

WHY DICTS NOT DATAFRAMES:
    The Gemini API passes tool results as JSON.  DataFrames and numpy arrays
    can't be serialised.  We convert analytics output to clean dicts/lists here
    so the LLM gets readable structured data.

WHY TOOLS CALL ANALYTICS FUNCTIONS:
    Tools don't re-implement any calculation logic.  They are thin wrappers:
        tool = DB query → analytics function → JSON dict
    This keeps analytics.py functions testable independently.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.db.queries import (
    get_player_deliveries,
    get_batter_bowler_deliveries,
    get_matches,
    search_players,
)
from src.analytics.batting import (
    get_batting_summary,
    get_phase_batting_stats,
    get_batting_season_trend,
    get_dismissal_breakdown,
)
from src.analytics.bowling import (
    get_bowling_summary,
    get_phase_bowling_stats,
    get_wicket_type_breakdown,
)
from src.analytics.matchup import get_matchup_summary, get_matchup_by_phase
from src.analytics.team import get_head_to_head, get_venue_summary, get_toss_impact
from src.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Player name resolver  (fuzzy search → DB id)
# ---------------------------------------------------------------------------

def _resolve_player(name: str) -> tuple[int | None, str | None]:
    """
    Resolve a player name to (player_id, canonical_name).

    Cricsheet stores abbreviated names like "JJ Bumrah", "B Kumar", "TA Boult".
    When the LLM passes a full name like "Jasprit Bumrah" we try two strategies:

    1. Full-name fuzzy search  — catches "V Kohli", "MS Dhoni"
    2. Surname-only search     — catches "Bumrah" → "JJ Bumrah",
                                 "Kumar" → "B Kumar", "Boult" → "TA Boult"

    This eliminates the need for the LLM to manually call search_player_name
    before every stats lookup, saving 1–2 tool-call rounds per player.
    """
    # Strategy 1: full name partial match
    results = search_players(name)
    if not results.empty:
        row = results.iloc[0]
        return int(row["id"]), str(row["name"])

    # Strategy 2: surname only (last word of the passed name)
    parts = name.strip().split()
    if len(parts) > 1:
        surname = parts[-1]
        if len(surname) > 3:          # skip trivial suffixes like "Jr", "de"
            results = search_players(surname)
            if not results.empty:
                row = results.iloc[0]
                return int(row["id"]), str(row["name"])

    return None, None


def _df_to_records(df: pd.DataFrame, max_rows: int = 20) -> list[dict]:
    """Convert a DataFrame to a list of dicts, rounding floats for readability."""
    if df.empty:
        return []
    df = df.head(max_rows).copy()
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = df[col].round(2)
    return df.fillna("—").to_dict(orient="records")


def _safe(value, fmt: str = "{:.2f}") -> str | None:
    """Format a numeric value safely; return None if value is None."""
    if value is None:
        return None
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


# ===========================================================================
# TOOL 1 — Batting stats
# ===========================================================================

def get_batting_stats(player_name: str, season: str | None = None) -> dict:
    """
    Get career batting statistics for an IPL player.

    Args:
        player_name: Player name (partial match supported, e.g. "Kohli")
        season:      IPL season string, e.g. "2023/24" (optional; None = career)

    Returns dict with: player_name, matches, innings, runs, avg, sr,
                       fours, sixes, fifties, hundreds, highest_score,
                       dot_pct, boundary_pct, phase_breakdown
    """
    pid, canonical = _resolve_player(player_name)
    if pid is None:
        return {"error": f"Player '{player_name}' not found in IPL database."}

    logger.info("[tool] get_batting_stats: player=%s id=%d season=%s", canonical, pid, season)
    df = get_player_deliveries(player_id=pid, role="batter")

    if season and not df.empty and "season" in df.columns:
        df = df[df["season"] == season]

    if df.empty:
        return {"error": f"No batting data for {canonical}" + (f" in {season}" if season else "")}

    summary   = get_batting_summary(df)
    phase_df  = get_phase_batting_stats(df)
    dismiss_df = get_dismissal_breakdown(df)

    phase_breakdown = []
    for _, row in phase_df.iterrows():
        phase_breakdown.append({
            "phase":       row.get("phase_label", row.get("phase", "")),
            "balls_faced": int(row.get("balls_faced", 0)),
            "runs":        int(row.get("runs", 0)),
            "sr":          _safe(row.get("sr")),
            "avg":         _safe(row.get("avg")),
            "dot_pct":     _safe(row.get("dot_pct")),
        })

    top_dismissals = _df_to_records(dismiss_df.head(3), max_rows=3)

    return {
        "player_name":    canonical,
        "season":         season or "career",
        "matches":        summary.get("matches"),
        "innings":        summary.get("innings"),
        "runs":           summary.get("runs"),
        "avg":            _safe(summary.get("avg")),
        "sr":             _safe(summary.get("sr")),
        "fours":          summary.get("fours"),
        "sixes":          summary.get("sixes"),
        "fifties":        summary.get("fifties"),
        "hundreds":       summary.get("hundreds"),
        "highest_score":  summary.get("highest_score"),
        "dot_pct":        _safe(summary.get("dot_pct")),
        "boundary_pct":   _safe(summary.get("boundary_pct")),
        "phase_breakdown": phase_breakdown,
        "top_dismissals":  top_dismissals,
    }


# ===========================================================================
# TOOL 2 — Bowling stats
# ===========================================================================

def get_bowling_stats(player_name: str, season: str | None = None) -> dict:
    """
    Get career bowling statistics for an IPL player.

    Args:
        player_name: Player name (partial match OK)
        season:      IPL season string (optional; None = career)

    Returns dict with: player_name, matches, overs, wickets, economy,
                       avg, sr, dot_pct, best_bowling, phase_breakdown
    """
    pid, canonical = _resolve_player(player_name)
    if pid is None:
        return {"error": f"Player '{player_name}' not found."}

    logger.info("[tool] get_bowling_stats: player=%s id=%d season=%s", canonical, pid, season)
    df = get_player_deliveries(player_id=pid, role="bowler")

    if season and not df.empty and "season" in df.columns:
        df = df[df["season"] == season]

    if df.empty:
        return {"error": f"No bowling data for {canonical}" + (f" in {season}" if season else "")}

    summary  = get_bowling_summary(df)
    phase_df = get_phase_bowling_stats(df)
    wkt_df   = get_wicket_type_breakdown(df)

    phase_breakdown = []
    for _, row in phase_df.iterrows():
        phase_breakdown.append({
            "phase":         row.get("phase_label", row.get("phase", "")),
            "overs":         _safe(row.get("overs"), "{:.1f}"),
            "wickets":       int(row.get("wickets", 0)),
            "economy":       _safe(row.get("economy")),
            "dot_pct":       _safe(row.get("dot_pct")),
        })

    return {
        "player_name":    canonical,
        "season":         season or "career",
        "matches":        summary.get("matches"),
        "overs":          _safe(summary.get("overs"), "{:.1f}"),
        "wickets":        summary.get("wickets"),
        "economy":        _safe(summary.get("economy")),
        "avg":            _safe(summary.get("avg")),
        "sr":             _safe(summary.get("sr")),
        "dot_pct":        _safe(summary.get("dot_pct")),
        "best_bowling":   summary.get("best_bowling"),
        "four_wkt_hauls": summary.get("four_wicket_hauls"),
        "five_wkt_hauls": summary.get("five_wicket_hauls"),
        "phase_breakdown": phase_breakdown,
        "wicket_types":   _df_to_records(wkt_df.head(4), max_rows=4),
    }


# ===========================================================================
# TOOL 3 — Batter vs Bowler matchup
# ===========================================================================

def get_matchup(batter_name: str, bowler_name: str) -> dict:
    """
    Get head-to-head statistics between a batter and a bowler.

    Args:
        batter_name: Batter's name
        bowler_name: Bowler's name

    Returns dict with: batter, bowler, balls_faced, runs, dismissals,
                       sr, economy, dot_pct, dominance_index, phase_breakdown
    """
    batter_id, batter_canonical = _resolve_player(batter_name)
    bowler_id, bowler_canonical = _resolve_player(bowler_name)

    if batter_id is None:
        return {"error": f"Batter '{batter_name}' not found."}
    if bowler_id is None:
        return {"error": f"Bowler '{bowler_name}' not found."}
    if batter_id == bowler_id:
        return {"error": "Batter and bowler must be different players."}

    logger.info("[tool] get_matchup: %s vs %s", batter_canonical, bowler_canonical)
    df = get_batter_bowler_deliveries(batter_id=batter_id, bowler_id=bowler_id)

    if df.empty:
        return {
            "batter": batter_canonical,
            "bowler": bowler_canonical,
            "error":  "These two players have never faced each other in the IPL database.",
        }

    summary  = get_matchup_summary(df)
    phase_df = get_matchup_by_phase(df)

    phase_breakdown = []
    for _, row in phase_df.iterrows():
        phase_breakdown.append({
            "phase":       row.get("phase_label", row.get("phase", "")),
            "balls":       int(row.get("balls_faced", 0)),
            "runs":        int(row.get("runs", 0)),
            "dismissals":  int(row.get("dismissals", 0)),
            "sr":          _safe(row.get("sr")),
            "economy":     _safe(row.get("economy")),
        })

    dom = summary.get("dominance_index") or 0
    if dom > 0.1:
        verdict = f"{batter_canonical} has the edge"
    elif dom < -0.1:
        verdict = f"{bowler_canonical} has the edge"
    else:
        verdict = "Evenly matched"

    return {
        "batter":          batter_canonical,
        "bowler":          bowler_canonical,
        "balls_faced":     summary.get("balls_faced"),
        "runs":            summary.get("runs"),
        "dismissals":      summary.get("dismissals"),
        "sr":              _safe(summary.get("sr")),
        "economy":         _safe(summary.get("economy")),
        "dot_pct":         _safe(summary.get("dot_pct")),
        "boundary_pct":    _safe(summary.get("boundary_pct")),
        "fours":           summary.get("fours"),
        "sixes":           summary.get("sixes"),
        "matches_met":     summary.get("matches_met"),
        "dominance_index": _safe(summary.get("dominance_index")),
        "verdict":         verdict,
        "phase_breakdown": phase_breakdown,
    }


# ===========================================================================
# TOOL 4 — Team head-to-head
# ===========================================================================

def get_team_head_to_head(team_a: str, team_b: str) -> dict:
    """
    Get head-to-head record between two IPL teams.

    Args:
        team_a: First team name (e.g. "Mumbai Indians")
        team_b: Second team name (e.g. "Chennai Super Kings")

    Returns dict with win counts, win percentages, total matches.
    """
    logger.info("[tool] get_team_head_to_head: %s vs %s", team_a, team_b)
    matches_df = get_matches(limit=2000).rename(columns={
        "team1": "team1_name", "team2": "team2_name",
        "winner": "winner_name", "venue": "venue_name",
    })
    h2h = get_head_to_head(matches_df, team_a, team_b)

    if h2h.get("total_matches", 0) == 0:
        return {"error": f"No matches found between '{team_a}' and '{team_b}'."}

    return {
        "team_a":           team_a,
        "team_b":           team_b,
        "total_matches":    h2h["total_matches"],
        "team_a_wins":      h2h["team_a_wins"],
        "team_b_wins":      h2h["team_b_wins"],
        "no_result":        h2h.get("no_result", 0),
        "team_a_win_pct":   _safe(h2h.get("team_a_win_pct"), "{:.1f}%"),
        "team_b_win_pct":   _safe(h2h.get("team_b_win_pct"), "{:.1f}%"),
    }


# ===========================================================================
# TOOL 5 — Win probability
# ===========================================================================

def predict_win_probability(
    runs_scored: int,
    wickets_fallen: int,
    balls_remaining: int,
    required_runs: int,
    phase: str,
) -> dict:
    """
    Predict win probability for the BATTING team in a T20 chase.

    Args:
        runs_scored:     Runs scored so far in the 2nd innings
        wickets_fallen:  Wickets lost (0–9)
        balls_remaining: Legal balls left (T20 max 120)
        required_runs:   Runs still needed to win
        phase:           'powerplay' | 'middle' | 'death'

    Returns: { batting_team_win_pct, bowling_team_win_pct, interpretation }
    """
    from src.ml.win_probability import WinProbabilityModel, MODEL_PATH
    if not MODEL_PATH.exists():
        return {"error": "Win probability model not trained. Run scripts/train_win_model.py first."}

    model = WinProbabilityModel.load()
    prob  = model.predict(
        runs_scored=runs_scored, wickets_fallen=wickets_fallen,
        balls_remaining=balls_remaining, required_runs=required_runs,
        phase=phase,
    )

    if prob >= 0.75:
        interp = "Batting team is in strong control"
    elif prob >= 0.55:
        interp = "Batting team has a slight edge"
    elif prob >= 0.45:
        interp = "Evenly balanced"
    elif prob >= 0.25:
        interp = "Bowling team has the advantage"
    else:
        interp = "Bowling team is in strong control"

    return {
        "batting_team_win_pct":  f"{prob:.1%}",
        "bowling_team_win_pct":  f"{1-prob:.1%}",
        "interpretation":        interp,
        "required_run_rate":     _safe(required_runs / (balls_remaining / 6) if balls_remaining > 0 else 36, "{:.2f}"),
        "current_run_rate":      _safe(runs_scored / ((120 - balls_remaining) / 6) if (120 - balls_remaining) > 0 else 0, "{:.2f}"),
    }


# ===========================================================================
# TOOL 6 — Monte Carlo score simulation
# ===========================================================================

def simulate_innings_score(
    balls_bowled: int = 0,
    runs_so_far: int = 0,
    wickets_fallen: int = 0,
    target: int | None = None,
    n_simulations: int = 5000,
) -> dict:
    """
    Simulate a T20 innings using historical IPL ball-outcome distributions.

    Args:
        balls_bowled:   Legal balls already bowled (0 = full innings)
        runs_so_far:    Runs already scored
        wickets_fallen: Wickets already lost (0–9)
        target:         Chase target (None for 1st innings)
        n_simulations:  Number of simulations (max 10000)

    Returns: { mean_score, median_score, p10, p90, prob_of_reaching_target }
    """
    from src.simulation.monte_carlo import MonteCarloSimulator
    from src.simulation.distributions import DISTRIBUTIONS_PATH
    if not DISTRIBUTIONS_PATH.exists():
        return {"error": "Ball distributions not computed. Run scripts/compute_distributions.py first."}

    sim = MonteCarloSimulator.load(seed=42)
    result = sim.simulate(
        n_simulations  = min(n_simulations, 10_000),
        balls_bowled   = balls_bowled,
        runs_so_far    = runs_so_far,
        wickets_fallen = wickets_fallen,
        target         = target,
    )

    out: dict = {
        "mean_score":     round(result.mean_score),
        "median_score":   round(result.median_score),
        "percentile_10":  round(result.percentile_10),
        "percentile_90":  round(result.percentile_90),
        "n_simulations":  result.n_simulations,
    }
    if target:
        out["prob_of_reaching_target"] = f"{result.probability_of_reaching(target):.1%}"
        out["target"] = target

    return out


# ===========================================================================
# TOOL 7 — Player search
# ===========================================================================

def search_player_name(query: str) -> dict:
    """
    Search for IPL players by name. Useful when unsure of the exact spelling.

    Args:
        query: Partial player name, e.g. "bumrah", "kohli", "dhoni"

    Returns: { matches: [ {id, name}, ... ] } (top 5 results)
    """
    results = search_players(query)
    if results.empty:
        return {"matches": [], "message": f"No players found matching '{query}'."}
    top5 = results.head(5)
    return {
        "matches": [
            {"id": int(r["id"]), "name": str(r["name"])}
            for _, r in top5.iterrows()
        ]
    }


# ===========================================================================
# TOOL 8 — Venue stats
# ===========================================================================

def get_venue_stats(venue_name: str) -> dict:
    """
    Get batting/bowling first win percentages and match count for a venue.

    Args:
        venue_name: Full or partial venue name, e.g. "Wankhede", "Chepauk"

    Returns: { venue_name, matches, bat_first_win_pct, bowl_first_win_pct, recommendation }
    """
    logger.info("[tool] get_venue_stats: %s", venue_name)
    matches_df = get_matches(limit=2000).rename(columns={
        "team1": "team1_name", "team2": "team2_name",
        "winner": "winner_name", "venue": "venue_name",
    })

    # Filter by partial venue name match
    filtered = matches_df[
        matches_df["venue_name"].str.contains(venue_name, case=False, na=False)
    ]
    if filtered.empty:
        return {"error": f"No matches found at venue matching '{venue_name}'."}

    venue_df = get_venue_summary(filtered)
    if venue_df.empty:
        return {"error": f"Could not compute venue stats for '{venue_name}'."}

    row = venue_df.iloc[0]
    bat_pct  = float(row.get("bat_first_win_pct") or 50)
    bowl_pct = 100 - bat_pct

    if bat_pct > 55:
        recommendation = "Bat first — this venue historically favours batting first"
    elif bat_pct < 45:
        recommendation = "Bowl first — this venue favours chasing"
    else:
        recommendation = "No strong toss preference — both strategies are equally viable"

    return {
        "venue_name":         str(row.get("venue_name", venue_name)),
        "matches":            int(row.get("matches", 0)),
        "bat_first_wins":     int(row.get("bat_first_wins", 0)),
        "bowl_first_wins":    int(row.get("bowl_first_wins", 0)),
        "bat_first_win_pct":  f"{bat_pct:.1f}%",
        "bowl_first_win_pct": f"{bowl_pct:.1f}%",
        "toss_recommendation": recommendation,
    }


# ===========================================================================
# Tool registry — used by agent.py to register Gemini function declarations
# ===========================================================================

TOOL_FUNCTIONS: dict[str, callable] = {
    "get_batting_stats":       get_batting_stats,
    "get_bowling_stats":       get_bowling_stats,
    "get_matchup":             get_matchup,
    "get_team_head_to_head":   get_team_head_to_head,
    "predict_win_probability": predict_win_probability,
    "simulate_innings_score":  simulate_innings_score,
    "search_player_name":      search_player_name,
    "get_venue_stats":         get_venue_stats,
}
