"""
CricketIQ — Batting Analytics
================================
All functions accept a deliveries DataFrame (from db.queries.get_player_deliveries)
and return a DataFrame or scalar.

COLUMN CONTRACT (input DataFrame):
    Required from get_player_deliveries():
        match_id, innings_id, over_number, ball_in_over,
        batter_name, bowler_name,
        runs_batter, runs_extras, runs_total,
        is_legal_delivery, is_boundary, is_six, is_dot_ball, is_wicket,
        player_out_id, batter_id, wicket_kind, phase,
        season (year string), match_date, venue_name

WHY DATAFRAMES IN, DATAFRAMES OUT:
    All analytics functions are pure transformations — they don't touch the
    database and don't know about Streamlit.  This makes them independently
    testable and reusable by the AI agent's tool layer.
"""

from __future__ import annotations

import pandas as pd

from src.analytics.metrics import (
    batting_average,
    boundary_percentage,
    dot_ball_percentage,
    overs_from_deliveries,
    phase_label,
    strike_rate,
)


# ---------------------------------------------------------------------------
# Career / overall summary
# ---------------------------------------------------------------------------

def get_batting_summary(df: pd.DataFrame) -> dict:
    """
    Compute a single-player batting summary across all deliveries in df.

    Returns a dict (not a DataFrame) because this is a single-row card of
    headline numbers — displayed as a stats panel in Streamlit, not a table.

    Keys: matches, innings, runs, balls_faced, avg, sr, fours, sixes,
          fifties, hundreds, highest_score, dot_pct, boundary_pct
    """
    if df.empty:
        return _empty_batting_summary()

    matches    = df["match_id"].nunique()
    innings    = df.groupby("match_id")["runs_batter"].sum().gt(0).sum()
    # A "proper" innings: batter faced at least 1 ball in that match
    innings    = df[df["is_legal_delivery"]]["match_id"].nunique()

    runs         = int(df["runs_batter"].sum())
    balls_faced  = int(df["is_legal_delivery"].sum())
    fours        = int((df["is_boundary"] & ~df["is_six"]).sum())
    sixes        = int(df["is_six"].sum())
    dot_balls    = int(df["is_dot_ball"].sum())

    # Dismissals: deliveries where this batter was the one out
    dismissals = int(
        (df["is_wicket"] & (df["player_out_id"] == df["batter_id"])).sum()
    )

    # Per-innings runs for fifties / hundreds / highest
    per_innings = (
        df.groupby("match_id")["runs_batter"].sum().reset_index(name="innings_runs")
    )
    fifties  = int((per_innings["innings_runs"].between(50, 99)).sum())
    hundreds = int((per_innings["innings_runs"] >= 100).sum())
    highest  = int(per_innings["innings_runs"].max()) if not per_innings.empty else 0

    return {
        "matches":       matches,
        "innings":       innings,
        "runs":          runs,
        "balls_faced":   balls_faced,
        "dismissals":    dismissals,
        "avg":           batting_average(runs, dismissals),
        "sr":            strike_rate(runs, balls_faced),
        "fours":         fours,
        "sixes":         sixes,
        "fifties":       fifties,
        "hundreds":      hundreds,
        "highest_score": highest,
        "dot_pct":       dot_ball_percentage(dot_balls, balls_faced),
        "boundary_pct":  boundary_percentage(fours + sixes, balls_faced),
    }


def _empty_batting_summary() -> dict:
    keys = [
        "matches", "innings", "runs", "balls_faced", "dismissals",
        "avg", "sr", "fours", "sixes", "fifties", "hundreds",
        "highest_score", "dot_pct", "boundary_pct",
    ]
    return {k: None for k in keys}


# ---------------------------------------------------------------------------
# Phase breakdown
# ---------------------------------------------------------------------------

def get_phase_batting_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split batting performance across powerplay / middle / death.

    Returns a DataFrame indexed by phase with columns:
        balls_faced, runs, dismissals, sr, dot_pct, boundary_pct, phase_label

    WHY: In T20s, knowing if a player struggles in the death vs dominates
    the powerplay is more actionable than a single career average.
    """
    if df.empty:
        return pd.DataFrame()

    def _phase_agg(grp: pd.DataFrame) -> pd.Series:
        balls    = int(grp["is_legal_delivery"].sum())
        runs     = int(grp["runs_batter"].sum())
        dism     = int(
            (grp["is_wicket"] & (grp["player_out_id"] == grp["batter_id"])).sum()
        )
        fours    = int((grp["is_boundary"] & ~grp["is_six"]).sum())
        sixes    = int(grp["is_six"].sum())
        dots     = int(grp["is_dot_ball"].sum())
        return pd.Series({
            "balls_faced":   balls,
            "runs":          runs,
            "dismissals":    dism,
            "fours":         fours,
            "sixes":         sixes,
            "sr":            strike_rate(runs, balls),
            "avg":           batting_average(runs, dism),
            "dot_pct":       dot_ball_percentage(dots, balls),
            "boundary_pct":  boundary_percentage(fours + sixes, balls),
        })

    result = df.groupby("phase").apply(_phase_agg, include_groups=False)

    # Ensure canonical phase order and add readable label
    phase_order = ["powerplay", "middle", "death"]
    result = result.reindex([p for p in phase_order if p in result.index])
    result["phase_label"] = result.index.map(phase_label)
    return result.reset_index()


# ---------------------------------------------------------------------------
# Season trend
# ---------------------------------------------------------------------------

def get_batting_season_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Year-on-year batting evolution: one row per season.

    Columns: season, matches, runs, balls_faced, dismissals, avg, sr,
             fours, sixes, boundary_pct

    WHY: Trend data lets us spot if a player is improving, declining, or
    consistent — and whether that correlates with age, team changes, etc.
    """
    if df.empty:
        return pd.DataFrame()

    def _season_agg(grp: pd.DataFrame) -> pd.Series:
        balls   = int(grp["is_legal_delivery"].sum())
        runs    = int(grp["runs_batter"].sum())
        dism    = int(
            (grp["is_wicket"] & (grp["player_out_id"] == grp["batter_id"])).sum()
        )
        fours   = int((grp["is_boundary"] & ~grp["is_six"]).sum())
        sixes   = int(grp["is_six"].sum())
        matches = grp["match_id"].nunique()
        return pd.Series({
            "matches":      matches,
            "runs":         runs,
            "balls_faced":  balls,
            "dismissals":   dism,
            "avg":          batting_average(runs, dism),
            "sr":           strike_rate(runs, balls),
            "fours":        fours,
            "sixes":        sixes,
            "boundary_pct": boundary_percentage(fours + sixes, balls),
        })

    result = df.groupby("season").apply(_season_agg, include_groups=False)
    return result.reset_index().sort_values("season")


# ---------------------------------------------------------------------------
# Over-by-over scoring map
# ---------------------------------------------------------------------------

def get_scoring_by_over(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average runs scored and strike rate for each over (1–20).

    Columns: over_number, avg_runs_per_over, sr, balls_faced

    WHY: Identifies which overs a batter is most productive in.
    A specialist finisher will spike in overs 18–20; an opener will be
    most active in 1–6.
    """
    if df.empty:
        return pd.DataFrame()

    legal = df[df["is_legal_delivery"]].copy()

    agg = (
        legal.groupby("over_number")
        .agg(
            balls_faced=("runs_batter", "count"),
            runs=("runs_batter", "sum"),
        )
        .reset_index()
    )
    agg["sr"] = agg.apply(
        lambda r: strike_rate(r["runs"], r["balls_faced"]), axis=1
    )
    # Normalise to per-over average (across however many times batter faced that over)
    n_matches = df["match_id"].nunique()
    agg["avg_runs_per_over"] = (agg["runs"] / n_matches).round(2)

    return agg.sort_values("over_number")


# ---------------------------------------------------------------------------
# Dismissal analysis
# ---------------------------------------------------------------------------

def get_dismissal_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    How often and how the player gets out.

    Columns: wicket_kind, count, pct

    WHY: A player who is repeatedly caught at deep midwicket has a specific
    weakness the AI agent can flag when recommending bowling plans.
    """
    wickets = df[
        df["is_wicket"] & (df["player_out_id"] == df["batter_id"])
    ].copy()

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
# Venue performance
# ---------------------------------------------------------------------------

def get_batting_by_venue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-venue batting summary for a player.

    Columns: venue_name, matches, runs, balls_faced, avg, sr

    WHY: Some players dominate at flat pitches (Wankhede, Chinnaswamy) and
    struggle on slower wickets (Chepauk).  This drives venue-specific selections.
    """
    if df.empty or "venue_name" not in df.columns:
        return pd.DataFrame()

    def _venue_agg(grp: pd.DataFrame) -> pd.Series:
        balls   = int(grp["is_legal_delivery"].sum())
        runs    = int(grp["runs_batter"].sum())
        dism    = int(
            (grp["is_wicket"] & (grp["player_out_id"] == grp["batter_id"])).sum()
        )
        return pd.Series({
            "matches":     grp["match_id"].nunique(),
            "runs":        runs,
            "balls_faced": balls,
            "avg":         batting_average(runs, dism),
            "sr":          strike_rate(runs, balls),
        })

    result = df.groupby("venue_name").apply(_venue_agg, include_groups=False)
    return result.reset_index().sort_values("runs", ascending=False)
