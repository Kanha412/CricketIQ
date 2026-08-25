"""
CricketIQ — Bowling Analytics
================================
All functions accept a deliveries DataFrame (from db.queries.get_player_deliveries
called with role='bowler') and return a DataFrame or dict.

COLUMN CONTRACT (input DataFrame):
    Same schema as batting.py but rows are for the bowler's deliveries:
        match_id, innings_id, over_number, ball_in_over,
        batter_name, bowler_name, bowler_id,
        runs_total, is_legal_delivery, is_wicket, wicket_kind,
        is_dot_ball, is_boundary, is_six, extras_type, phase, season
"""

from __future__ import annotations

import pandas as pd

from src.analytics.metrics import (
    bowling_average,
    bowling_dot_percentage,
    bowling_strike_rate,
    economy_rate,
    overs_from_deliveries,
    phase_label,
    strike_rate,
)


# ---------------------------------------------------------------------------
# Career / overall summary
# ---------------------------------------------------------------------------

def get_bowling_summary(df: pd.DataFrame) -> dict:
    """
    Compute a single-player bowling summary across all deliveries in df.

    Returns a dict with headline stats for the stats-card panel.
    Keys: matches, overs, runs_conceded, wickets, economy, avg, sr, dot_pct,
          best_bowling (str), four_wicket_hauls, five_wicket_hauls
    """
    if df.empty:
        return _empty_bowling_summary()

    matches     = int(df["match_id"].nunique())
    legal       = int(df["is_legal_delivery"].sum())
    runs        = int(df["runs_total"].sum())
    wickets     = int(df["is_wicket"].sum())
    dot_balls   = int(df["is_dot_ball"].sum())

    # Per-match wicket tallies for best bowling / hauls
    per_match = (
        df.groupby("match_id")
        .agg(
            match_wickets=("is_wicket", "sum"),
            match_runs=("runs_total", "sum"),
        )
        .reset_index()
    )
    four_wkt  = int((per_match["match_wickets"] >= 4).sum())
    five_wkt  = int((per_match["match_wickets"] >= 5).sum())

    # Best bowling: most wickets, then fewest runs on tied wickets
    best_row = per_match.sort_values(
        ["match_wickets", "match_runs"], ascending=[False, True]
    ).iloc[0] if not per_match.empty else None
    best_str = (
        f"{int(best_row['match_wickets'])}/{int(best_row['match_runs'])}"
        if best_row is not None else None
    )

    return {
        "matches":           matches,
        "overs":             overs_from_deliveries(legal),
        "runs_conceded":     runs,
        "wickets":           wickets,
        "economy":           economy_rate(runs, legal),
        "avg":               bowling_average(runs, wickets),
        "sr":                bowling_strike_rate(legal, wickets),
        "dot_pct":           bowling_dot_percentage(dot_balls, legal),
        "four_wicket_hauls": four_wkt,
        "five_wicket_hauls": five_wkt,
        "best_bowling":      best_str,
    }


def _empty_bowling_summary() -> dict:
    keys = [
        "matches", "overs", "runs_conceded", "wickets",
        "economy", "avg", "sr", "dot_pct",
        "four_wicket_hauls", "five_wicket_hauls", "best_bowling",
    ]
    return {k: None for k in keys}


# ---------------------------------------------------------------------------
# Phase breakdown
# ---------------------------------------------------------------------------

def get_phase_bowling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Economy, wickets, and dot-ball % split by powerplay / middle / death.

    Returns DataFrame with columns:
        phase, overs, runs_conceded, wickets, economy, dot_pct, phase_label

    WHY: A pacer might have great powerplay economy but get smashed in the
    death — this tells the AI agent to flag that in bowling plans.
    """
    if df.empty:
        return pd.DataFrame()

    def _phase_agg(grp: pd.DataFrame) -> pd.Series:
        legal   = int(grp["is_legal_delivery"].sum())
        runs    = int(grp["runs_total"].sum())
        wickets = int(grp["is_wicket"].sum())
        dots    = int(grp["is_dot_ball"].sum())
        return pd.Series({
            "overs":         overs_from_deliveries(legal),
            "runs_conceded": runs,
            "wickets":       wickets,
            "economy":       economy_rate(runs, legal),
            "avg":           bowling_average(runs, wickets),
            "dot_pct":       bowling_dot_percentage(dots, legal),
        })

    result = df.groupby("phase").apply(_phase_agg, include_groups=False)
    phase_order = ["powerplay", "middle", "death"]
    result = result.reindex([p for p in phase_order if p in result.index])
    result["phase_label"] = result.index.map(phase_label)
    return result.reset_index()


# ---------------------------------------------------------------------------
# Over-by-over economy
# ---------------------------------------------------------------------------

def get_economy_by_over(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average economy and wickets taken in each over number (1–20).

    Columns: over_number, avg_runs_per_over, wickets, dot_pct, economy

    WHY: Death specialists (overs 17–20) can have high economy in early
    overs but that's acceptable — context matters.
    """
    if df.empty:
        return pd.DataFrame()

    legal = df[df["is_legal_delivery"]].copy()
    n_matches = df["match_id"].nunique()

    agg = (
        legal.groupby("over_number")
        .agg(
            deliveries=("runs_total", "count"),
            runs=("runs_total", "sum"),
            wickets=("is_wicket", "sum"),
            dots=("is_dot_ball", "sum"),
        )
        .reset_index()
    )
    agg["economy"]          = agg.apply(lambda r: economy_rate(r["runs"], r["deliveries"]), axis=1)
    agg["dot_pct"]          = agg.apply(lambda r: bowling_dot_percentage(r["dots"], r["deliveries"]), axis=1)
    agg["avg_runs_per_over"] = (agg["runs"] / n_matches).round(2)

    return agg.sort_values("over_number")


# ---------------------------------------------------------------------------
# Season trend
# ---------------------------------------------------------------------------

def get_bowling_season_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Year-by-year bowling performance.

    Columns: season, matches, overs, runs_conceded, wickets, economy, avg, dot_pct
    """
    if df.empty:
        return pd.DataFrame()

    def _season_agg(grp: pd.DataFrame) -> pd.Series:
        legal   = int(grp["is_legal_delivery"].sum())
        runs    = int(grp["runs_total"].sum())
        wickets = int(grp["is_wicket"].sum())
        dots    = int(grp["is_dot_ball"].sum())
        return pd.Series({
            "matches":       grp["match_id"].nunique(),
            "overs":         overs_from_deliveries(legal),
            "runs_conceded": runs,
            "wickets":       wickets,
            "economy":       economy_rate(runs, legal),
            "avg":           bowling_average(runs, wickets),
            "dot_pct":       bowling_dot_percentage(dots, legal),
        })

    result = df.groupby("season").apply(_season_agg, include_groups=False)
    return result.reset_index().sort_values("season")


# ---------------------------------------------------------------------------
# Wicket type breakdown
# ---------------------------------------------------------------------------

def get_wicket_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distribution of dismissal types taken by this bowler.

    Columns: wicket_kind, count, pct

    WHY: Tells us if a bowler beats the bat (bowled/lbw) or induces catches.
    Bowled/LBW-heavy bowlers get more dangerous on helpful pitches.
    """
    wickets = df[df["is_wicket"]].copy()
    if wickets.empty:
        return pd.DataFrame(columns=["wicket_kind", "count", "pct"])

    breakdown = (
        wickets.groupby("wicket_kind")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    total = breakdown["count"].sum()
    breakdown["pct"] = (breakdown["count"] / total * 100).round(1)
    return breakdown


# ---------------------------------------------------------------------------
# Batter-type analysis (who does this bowler target)
# ---------------------------------------------------------------------------

def get_batter_type_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Economy and wickets against each batter who faced this bowler.

    Columns: batter_name, balls_faced, runs_conceded, wickets, economy

    WHY: Useful for match-up planning — "Jasprit Bumrah has an exceptional
    record against left-handers in the death."
    """
    if df.empty or "batter_name" not in df.columns:
        return pd.DataFrame()

    legal = df[df["is_legal_delivery"]]

    agg = (
        legal.groupby("batter_name")
        .agg(
            balls_faced=("runs_total", "count"),
            runs_conceded=("runs_total", "sum"),
            wickets=("is_wicket", "sum"),
        )
        .reset_index()
    )
    agg["economy"] = agg.apply(
        lambda r: economy_rate(r["runs_conceded"], r["balls_faced"]), axis=1
    )
    return agg.sort_values("balls_faced", ascending=False)
