"""
CricketIQ — Team & Venue Chart Functions
==========================================
All functions: (DataFrame | dict) → matplotlib.Figure
Inputs from src.analytics.team functions.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.style import (
    PALETTE,
    PHASE_COLORS,
    bar_labels,
    empty_chart,
    make_figure,
)


# ---------------------------------------------------------------------------
# Team win-rate — horizontal bar
# ---------------------------------------------------------------------------
def win_rate_chart(
    win_df: pd.DataFrame,
    title: str = "IPL Team Win Rates",
) -> matplotlib.figure.Figure:
    """
    Horizontal bar chart of win % for every team, coloured by how far above /
    below the 50% parity line they are.

    Input: output of analytics.team.get_team_win_rates()
    Columns: team, win_pct
    """
    if win_df.empty or "win_pct" not in win_df.columns:
        return empty_chart("No team win data available")

    df = win_df.sort_values("win_pct", ascending=True).copy()
    colors = [
        PALETTE["positive"] if w >= 50 else PALETTE["negative"]
        for w in df["win_pct"]
    ]

    fig, ax = make_figure(width=9, height=max(4, len(df) * 0.45 + 1))
    bars = ax.barh(df["team"], df["win_pct"], color=colors, alpha=0.85)

    # 50% parity reference
    ax.axvline(x=50, linestyle="--", color=PALETTE["neutral"], linewidth=1.2, label="50% parity")

    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{row['win_pct']:.1f}%",
            va="center", ha="left", fontsize=9, color=PALETTE["text"],
        )

    ax.set_xlabel("Win %")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, 80)
    ax.legend(framealpha=0)

    plt.close("all")
    return fig


# ---------------------------------------------------------------------------
# Head-to-head dumbbell
# ---------------------------------------------------------------------------
def head_to_head_chart(
    h2h: dict,
    team_a: str = "Team A",
    team_b: str = "Team B",
) -> matplotlib.figure.Figure:
    """
    Horizontal split bar showing each team's share of wins.

    Input: output of analytics.team.get_head_to_head()
    Keys: total_matches, team_a_wins, team_b_wins, no_result
    """
    if not h2h or h2h.get("total_matches", 0) == 0:
        return empty_chart(f"No matches found between {team_a} and {team_b}")

    a_wins = h2h.get("team_a_wins", 0)
    b_wins = h2h.get("team_b_wins", 0)
    nr     = h2h.get("no_result", 0)
    total  = h2h.get("total_matches", 1)

    fig, ax = make_figure(width=8, height=2.5)
    ax.barh([""], [a_wins / total * 100],         color=PALETTE["primary"],   alpha=0.85, label=team_a)
    ax.barh([""], [b_wins / total * 100],         color=PALETTE["secondary"], alpha=0.85,
            left=[a_wins / total * 100], label=team_b)
    ax.barh([""], [nr     / total * 100],         color=PALETTE["neutral"],   alpha=0.5,
            left=[(a_wins + b_wins) / total * 100], label="No Result")

    # Win count annotations
    if a_wins > 0:
        ax.text(a_wins / total * 50, 0, f"{a_wins}",
                ha="center", va="center", fontsize=16, fontweight="bold", color="white")
    if b_wins > 0:
        ax.text(a_wins / total * 100 + b_wins / total * 50, 0, f"{b_wins}",
                ha="center", va="center", fontsize=16, fontweight="bold", color=PALETTE["text"])

    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of matches (%)")
    ax.set_title(f"{team_a}  vs  {team_b}  ({total} matches)", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0)
    ax.set_yticks([])

    plt.close("all")
    return fig


# ---------------------------------------------------------------------------
# Toss impact — grouped bar
# ---------------------------------------------------------------------------
def toss_impact_chart(
    toss_data: dict,
    title: str = "Toss Impact Analysis",
) -> matplotlib.figure.Figure:
    """
    Grouped bar chart: win % when batting first vs fielding first after winning toss.

    Input: output of analytics.team.get_toss_impact()
    Keys: toss_winner_won_match_pct, bat_first_win_pct, field_first_win_pct
    """
    if not toss_data:
        return empty_chart("No toss data available")

    labels = ["Overall\n(toss winner)", "Chose to bat", "Chose to field"]
    values = [
        toss_data.get("toss_winner_won_match_pct") or 50,
        toss_data.get("bat_first_win_pct")         or 50,
        toss_data.get("field_first_win_pct")       or 50,
    ]
    colors = [
        PALETTE["primary"],
        PALETTE["positive"] if (values[1] or 50) >= 50 else PALETTE["negative"],
        PALETTE["positive"] if (values[2] or 50) >= 50 else PALETTE["negative"],
    ]

    fig, ax = make_figure(width=7, height=4)
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
    ax.axhline(y=50, linestyle="--", color=PALETTE["neutral"], linewidth=1, label="50% parity")
    bar_labels(ax, fmt="{:.1f}%", padding=0.5)
    ax.set_ylabel("Win %")
    ax.set_ylim(0, 90)
    ax.set_title(title, fontweight="bold")
    ax.legend(framealpha=0)

    plt.close("all")
    return fig


# ---------------------------------------------------------------------------
# Venue — bat first vs bowl first win % heatmap-style bar
# ---------------------------------------------------------------------------
def venue_win_rate_chart(
    venue_df: pd.DataFrame,
    top_n: int = 10,
) -> matplotlib.figure.Figure:
    """
    Horizontal grouped bar: bat-first win % vs bowl-first win % per venue.

    Input: output of analytics.team.get_venue_summary()
    Columns: venue_name, matches, bat_first_win_pct

    Shows only top_n venues by match count to avoid clutter.
    """
    if venue_df.empty or "bat_first_win_pct" not in venue_df.columns:
        return empty_chart("No venue data available")

    df = (
        venue_df
        .dropna(subset=["bat_first_win_pct"])
        .sort_values("matches", ascending=False)
        .head(top_n)
        .copy()
    )
    df["bowl_first_win_pct"] = 100 - df["bat_first_win_pct"]

    fig, ax = make_figure(width=10, height=max(4, len(df) * 0.5 + 1))
    y     = np.arange(len(df))
    width = 0.35

    ax.barh(y + width / 2, df["bat_first_win_pct"],  width, color=PALETTE["primary"],   alpha=0.85, label="Bat first wins %")
    ax.barh(y - width / 2, df["bowl_first_win_pct"], width, color=PALETTE["secondary"], alpha=0.85, label="Bowl first wins %")
    ax.axvline(x=50, linestyle="--", color=PALETTE["neutral"], linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df["venue_name"].tolist(), fontsize=9)
    ax.set_xlabel("Win %")
    ax.set_title(f"Venue Batting vs Bowling First Win % (top {top_n})", fontweight="bold")
    ax.legend(framealpha=0)
    ax.invert_yaxis()

    plt.close("all")
    return fig


# ---------------------------------------------------------------------------
# Phase scoring pattern — multi-team or single innings
# ---------------------------------------------------------------------------
def phase_scoring_chart(
    phase_df: pd.DataFrame,
    title: str = "Average Runs by Phase",
) -> matplotlib.figure.Figure:
    """
    Bar chart of average runs per phase with run-rate annotation.

    Input: output of analytics.team.get_innings_scoring_by_phase()
    Columns: phase, avg_runs, run_rate
    """
    if phase_df.empty or "avg_runs" not in phase_df.columns:
        return empty_chart("No phase scoring data")

    df = phase_df.copy()
    phase_order = ["powerplay", "middle", "death"]
    df["phase"] = pd.Categorical(df["phase"], categories=phase_order, ordered=True)
    df = df.sort_values("phase")

    colors = [PHASE_COLORS.get(p, PALETTE["primary"]) for p in df["phase"]]

    fig, ax = make_figure(width=8, height=4.5)
    bars = ax.bar(df["phase"], df["avg_runs"], color=colors, alpha=0.85, width=0.5)

    # Annotate with run-rate
    for bar, (_, row) in zip(bars, df.iterrows()):
        rr = row.get("run_rate")
        if rr and rr == rr:  # not NaN
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"RR: {rr:.1f}",
                ha="center", va="bottom", fontsize=9, color=PALETTE["text"],
            )

    ax.set_xlabel("Phase")
    ax.set_ylabel("Avg Runs")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(["Powerplay\n(1–6)", "Middle\n(7–15)", "Death\n(16–20)"])

    plt.close("all")
    return fig
