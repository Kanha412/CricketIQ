"""
CricketIQ — Batter vs Bowler Matchup Analytics
================================================
Head-to-head analysis between a specific batter and bowler.

Input: deliveries DataFrame from db.queries.get_batter_bowler_deliveries()
       (already pre-filtered to only that batter/bowler pair)

This module powers:
 - The "Matchup Analysis" section of each player profile
 - The AI agent's tactical commentary tool
 - Bowling-plan recommendations
"""

from __future__ import annotations

import pandas as pd

from src.analytics.metrics import (
    batting_average,
    bowling_average,
    boundary_percentage,
    dot_ball_percentage,
    economy_rate,
    phase_label,
    strike_rate,
)


# ---------------------------------------------------------------------------
# Overall matchup summary
# ---------------------------------------------------------------------------

def get_matchup_summary(df: pd.DataFrame) -> dict:
    """
    One-dict summary for a batter vs bowler matchup.

    WHY A DICT: The Streamlit UI and the AI agent both consume this as a
    "headline card" — it doesn't need to be a table, just key numbers.

    Returns keys:
        balls_faced, runs, dismissals, sr, economy,
        dot_pct, boundary_pct, fours, sixes, matches_met
    """
    if df.empty:
        return _empty_matchup()

    legal       = df[df["is_legal_delivery"]]
    balls       = int(legal.shape[0])
    runs        = int(legal["runs_batter"].sum())
    runs_total  = int(legal["runs_total"].sum())
    dismissals  = int(
        (df["is_wicket"] & (df["player_out_id"] == df["batter_id"])).sum()
    )
    fours   = int((legal["is_boundary"] & ~legal["is_six"]).sum())
    sixes   = int(legal["is_six"].sum())
    dots    = int(legal["is_dot_ball"].sum())

    return {
        "balls_faced":   balls,
        "runs":          runs,
        "dismissals":    dismissals,
        "sr":            strike_rate(runs, balls),
        "economy":       economy_rate(runs_total, balls),
        "dot_pct":       dot_ball_percentage(dots, balls),
        "boundary_pct":  boundary_percentage(fours + sixes, balls),
        "fours":         fours,
        "sixes":         sixes,
        "matches_met":   df["match_id"].nunique(),
        # Dominance signal: >1.0 means batter winning, <1.0 bowler winning
        "dominance_index": _dominance_index(runs, balls, dismissals),
    }


def _empty_matchup() -> dict:
    keys = [
        "balls_faced", "runs", "dismissals", "sr", "economy",
        "dot_pct", "boundary_pct", "fours", "sixes",
        "matches_met", "dominance_index",
    ]
    return {k: None for k in keys}


def _dominance_index(runs: int, balls: int, dismissals: int) -> float | None:
    """
    A single number summarising who has the upper hand.

    dominance = (actual_SR - 100) / 100  –  wicket_rate

    Where:
        actual_SR      = runs / balls * 100  (e.g. 50 means SR of 50)
        baseline SR    = 100 (dot-ball parity — neither side wins)
        wicket_rate    = dismissals / balls  (wickets per ball)

    Positive → batter advantage. Negative → bowler advantage.
    Returns None for 0 balls (no data).

    WHY: Subtracting the SR baseline of 100 means a batter hitting at SR 50
    is already -0.5 before adding wicket pressure — correctly flagging bowler
    dominance even when absolute run values look similar to batter-winning cases.
    The AI agent uses this to say "Kohli dominates Malinga (+0.42)".
    """
    if balls == 0:
        return None
    actual_sr    = (runs / balls) * 100     # strike rate
    sr_delta     = (actual_sr - 100) / 100  # deviation from neutral baseline
    wicket_rate  = dismissals / balls       # wickets per ball
    return round(sr_delta - wicket_rate, 3)


# ---------------------------------------------------------------------------
# Phase-level breakdown
# ---------------------------------------------------------------------------

def get_matchup_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Matchup performance split by phase.

    Columns: phase, balls_faced, runs, dismissals, sr, economy, dot_pct, phase_label

    WHY: A bowler who dominates a batter in the powerplay but gets hit in
    the death needs different handling — this surfaces that.
    """
    if df.empty:
        return pd.DataFrame()

    def _phase_agg(grp: pd.DataFrame) -> pd.Series:
        legal      = grp[grp["is_legal_delivery"]]
        balls      = int(legal.shape[0])
        runs       = int(legal["runs_batter"].sum())
        runs_total = int(legal["runs_total"].sum())
        dism       = int(
            (grp["is_wicket"] & (grp["player_out_id"] == grp["batter_id"])).sum()
        )
        dots  = int(legal["is_dot_ball"].sum())
        fours = int((legal["is_boundary"] & ~legal["is_six"]).sum())
        sixes = int(legal["is_six"].sum())
        return pd.Series({
            "balls_faced":  balls,
            "runs":         runs,
            "dismissals":   dism,
            "sr":           strike_rate(runs, balls),
            "economy":      economy_rate(runs_total, balls),
            "dot_pct":      dot_ball_percentage(dots, balls),
            "boundary_pct": boundary_percentage(fours + sixes, balls),
        })

    result = df.groupby("phase").apply(_phase_agg, include_groups=False)
    phase_order = ["powerplay", "middle", "death"]
    result = result.reindex([p for p in phase_order if p in result.index])
    result["phase_label"] = result.index.map(phase_label)
    return result.reset_index()


# ---------------------------------------------------------------------------
# Delivery-type / over distribution
# ---------------------------------------------------------------------------

def get_matchup_by_over(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs scored and wickets taken in each over number across all meetings.

    Columns: over_number, balls, runs, dismissals, sr

    WHY: Shows if a bowler gets the batter early in their spell (first 2 overs)
    or only after settling in — relevant for asking "should he bowl the first over?"
    """
    if df.empty:
        return pd.DataFrame()

    legal = df[df["is_legal_delivery"]]
    agg = (
        legal.groupby("over_number")
        .agg(
            balls=("runs_batter", "count"),
            runs=("runs_batter", "sum"),
            dismissals=("is_wicket", "sum"),
        )
        .reset_index()
    )
    agg["sr"] = agg.apply(lambda r: strike_rate(r["runs"], r["balls"]), axis=1)
    return agg.sort_values("over_number")


# ---------------------------------------------------------------------------
# Recent form (last N matches)
# ---------------------------------------------------------------------------

def get_matchup_recent_form(df: pd.DataFrame, last_n: int = 5) -> pd.DataFrame:
    """
    Matchup performance in the last N matches where they met.

    Columns: match_id, match_date, balls, runs, dismissed, sr

    WHY: Career stats can hide a dramatic recent form shift.  "In the last 5
    meetings, Kohli averages 8 against Bumrah" is more actionable than career.
    """
    if df.empty or "match_date" not in df.columns:
        return pd.DataFrame()

    recent_matches = (
        df[["match_id", "match_date"]]
        .drop_duplicates()
        .sort_values("match_date", ascending=False)
        .head(last_n)["match_id"]
        .tolist()
    )
    recent_df = df[df["match_id"].isin(recent_matches)]

    def _match_agg(grp: pd.DataFrame) -> pd.Series:
        legal = grp[grp["is_legal_delivery"]]
        balls = int(legal.shape[0])
        runs  = int(legal["runs_batter"].sum())
        dism  = int(
            (grp["is_wicket"] & (grp["player_out_id"] == grp["batter_id"])).sum()
        )
        date  = grp["match_date"].iloc[0] if "match_date" in grp.columns else None
        return pd.Series({
            "match_date": date,
            "balls":      balls,
            "runs":       runs,
            "dismissed":  bool(dism > 0),
            "sr":         strike_rate(runs, balls),
        })

    result = recent_df.groupby("match_id").apply(_match_agg, include_groups=False)
    return result.reset_index().sort_values("match_date", ascending=False)
