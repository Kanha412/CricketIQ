"""
CricketIQ — Team & Venue Analytics
=====================================
Functions that analyse team-level performance using the matches DataFrame
(from db.queries.get_matches) and the deliveries DataFrame.

These power:
 - The "Team Intelligence" dashboard page
 - The AI agent's venue-advice and opponent-analysis tools
"""

from __future__ import annotations

import pandas as pd

from src.analytics.metrics import economy_rate, run_rate, strike_rate


# ---------------------------------------------------------------------------
# Team win-rate
# ---------------------------------------------------------------------------

def get_team_win_rates(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Win rate for every team in the matches DataFrame.

    Input: matches DataFrame with columns:
        team1_name, team2_name, winner_name

    Returns: DataFrame with columns: team, played, won, win_pct
             sorted by win_pct descending.

    WHY: Simple but foundational — tells us which teams consistently win
    and becomes the baseline for adjusting head-to-head predictions.
    """
    if matches_df.empty:
        return pd.DataFrame(columns=["team", "played", "won", "win_pct"])

    # Each match contributes to two teams' played count
    team1 = matches_df[["team1_name", "winner_name"]].rename(columns={"team1_name": "team"})
    team2 = matches_df[["team2_name", "winner_name"]].rename(columns={"team2_name": "team"})
    all_teams = pd.concat([team1, team2], ignore_index=True)

    agg = (
        all_teams.groupby("team")
        .apply(
            # g.name is the group key (team name); with include_groups=False the
            # "team" column is excluded from g, so we must use g.name here.
            lambda g: pd.Series({
                "played": len(g),
                "won":    (g["winner_name"] == g.name).sum(),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    agg["win_pct"] = (agg["won"] / agg["played"] * 100).round(1)
    return agg.sort_values("win_pct", ascending=False)


# ---------------------------------------------------------------------------
# Head-to-head between two teams
# ---------------------------------------------------------------------------

def get_head_to_head(matches_df: pd.DataFrame, team_a: str, team_b: str) -> dict:
    """
    Head-to-head record between team_a and team_b.

    Returns a dict with:
        total_matches, team_a_wins, team_b_wins, no_result,
        team_a_win_pct, team_b_win_pct

    WHY: H2H matters more than overall win rate in tournament contexts —
    some teams have psychological edges over specific opponents.
    """
    h2h = matches_df[
        (
            (matches_df["team1_name"] == team_a) & (matches_df["team2_name"] == team_b)
        ) | (
            (matches_df["team1_name"] == team_b) & (matches_df["team2_name"] == team_a)
        )
    ].copy()

    if h2h.empty:
        return {
            "total_matches": 0,
            "team_a_wins": 0, "team_b_wins": 0, "no_result": 0,
            "team_a_win_pct": None, "team_b_win_pct": None,
        }

    total       = len(h2h)
    a_wins      = int((h2h["winner_name"] == team_a).sum())
    b_wins      = int((h2h["winner_name"] == team_b).sum())
    no_result   = total - a_wins - b_wins

    return {
        "total_matches":  total,
        "team_a_wins":    a_wins,
        "team_b_wins":    b_wins,
        "no_result":      no_result,
        "team_a_win_pct": round(a_wins / total * 100, 1) if total > 0 else None,
        "team_b_win_pct": round(b_wins / total * 100, 1) if total > 0 else None,
    }


# ---------------------------------------------------------------------------
# Toss analysis
# ---------------------------------------------------------------------------

def get_toss_impact(matches_df: pd.DataFrame) -> dict:
    """
    Does winning the toss actually help?

    Returns: { toss_winner_won_match_pct, bat_first_win_pct, field_first_win_pct }

    WHY: In T20s the conventional wisdom ("bowl first, chase") changes with
    venue and conditions.  Data-driven toss analysis informs captaincy decisions.
    """
    if matches_df.empty or "toss_winner" not in matches_df.columns:
        return {}

    valid = matches_df.dropna(subset=["toss_winner", "winner_name"])
    if valid.empty:
        return {}

    toss_won_match = (valid["toss_winner"] == valid["winner_name"]).sum()
    toss_total     = len(valid)

    bat_first  = valid[valid["toss_decision"] == "bat"]
    field_first = valid[valid["toss_decision"] == "field"]

    bat_wins   = (bat_first["toss_winner"] == bat_first["winner_name"]).sum()
    field_wins = (field_first["toss_winner"] == field_first["winner_name"]).sum()

    return {
        "toss_winner_won_match_pct": round(toss_won_match / toss_total * 100, 1),
        "bat_first_win_pct":  round(bat_wins / len(bat_first) * 100, 1) if len(bat_first) > 0 else None,
        "field_first_win_pct": round(field_wins / len(field_first) * 100, 1) if len(field_first) > 0 else None,
        "total_matches_analysed": toss_total,
    }


# ---------------------------------------------------------------------------
# Venue analytics
# ---------------------------------------------------------------------------

def get_venue_summary(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-venue match statistics.

    Columns: venue_name, matches, avg_first_innings_score,
             batting_first_wins, bowling_first_wins, bat_first_win_pct

    WHY: Different venues play completely differently.  Wankhede favours
    chasing; Chepauk (dry pitch) often favours batting first.  This data
    directly informs team selection strategy.
    """
    if matches_df.empty or "venue_name" not in matches_df.columns:
        return pd.DataFrame()

    def _venue_agg(grp: pd.DataFrame) -> pd.Series:
        matches        = len(grp)
        bat_first_wins = int(
            (
                (grp["toss_decision"] == "bat") & (grp["toss_winner"] == grp["winner_name"])
                | (grp["toss_decision"] == "field") & (grp["toss_winner"] != grp["winner_name"])
            ).sum()
        )
        bowl_first_wins = matches - bat_first_wins - int(grp["winner_name"].isna().sum())
        return pd.Series({
            "matches":           matches,
            "bat_first_wins":    bat_first_wins,
            "bowl_first_wins":   bowl_first_wins,
            "bat_first_win_pct": round(bat_first_wins / matches * 100, 1) if matches > 0 else None,
        })

    result = (
        matches_df.groupby("venue_name")
        .apply(_venue_agg, include_groups=False)
        .reset_index()
        .sort_values("matches", ascending=False)
    )
    return result


# ---------------------------------------------------------------------------
# Innings-level scoring patterns
# ---------------------------------------------------------------------------

def get_innings_scoring_by_phase(deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average runs scored in each phase across all matches in deliveries_df.

    Columns: phase, avg_runs, avg_wickets, run_rate

    Input: any deliveries DataFrame (can be filtered to a team, venue, season).

    WHY: "Teams average 58 runs in the powerplay at Wankhede" tells us if
    a venue is batting-friendly early on — direct input to game-plan advice.
    """
    if deliveries_df.empty:
        return pd.DataFrame()

    agg_per_match_phase = (
        deliveries_df.groupby(["match_id", "innings_id", "phase"])
        .agg(
            runs=("runs_total", "sum"),
            wickets=("is_wicket", "sum"),
            legal_balls=("is_legal_delivery", "sum"),
        )
        .reset_index()
    )

    phase_summary = (
        agg_per_match_phase.groupby("phase")
        .agg(
            avg_runs=("runs", "mean"),
            avg_wickets=("wickets", "mean"),
            avg_legal_balls=("legal_balls", "mean"),
        )
        .reset_index()
    )
    phase_summary["run_rate"] = phase_summary.apply(
        lambda r: economy_rate(r["avg_runs"], r["avg_legal_balls"]), axis=1
    )

    # Round for display
    phase_summary["avg_runs"]    = phase_summary["avg_runs"].round(1)
    phase_summary["avg_wickets"] = phase_summary["avg_wickets"].round(2)

    phase_order = ["powerplay", "middle", "death"]
    phase_summary["phase"] = pd.Categorical(phase_summary["phase"], categories=phase_order, ordered=True)
    return phase_summary.sort_values("phase")


# ---------------------------------------------------------------------------
# Season-by-season team performance
# ---------------------------------------------------------------------------

def get_team_season_record(matches_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """
    A team's wins/losses per season.

    Columns: season, played, won, lost, win_pct

    WHY: Shows dynasty cycles — MI's dominance in certain eras, CSK's
    consistency, etc. — useful for long-term tactical context.
    """
    if matches_df.empty:
        return pd.DataFrame()

    team_matches = matches_df[
        (matches_df["team1_name"] == team_name) | (matches_df["team2_name"] == team_name)
    ].copy()

    if team_matches.empty:
        return pd.DataFrame()

    def _season_agg(grp: pd.DataFrame) -> pd.Series:
        played = len(grp)
        won    = int((grp["winner_name"] == team_name).sum())
        return pd.Series({"played": played, "won": won, "lost": played - won})

    result = (
        team_matches.groupby("season")
        .apply(_season_agg, include_groups=False)
        .reset_index()
    )
    result["win_pct"] = (result["won"] / result["played"] * 100).round(1)
    return result.sort_values("season")
