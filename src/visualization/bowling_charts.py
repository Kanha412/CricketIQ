"""
CricketIQ — Bowling Chart Functions
======================================
All functions: (DataFrame | dict) → matplotlib.Figure
Inputs from src.analytics.bowling functions.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.style import (
    PALETTE,
    PHASE_COLORS,
    add_subtitle,
    bar_labels,
    empty_chart,
    make_figure,
)


# ---------------------------------------------------------------------------
# Phase bowling breakdown — grouped bar
# ---------------------------------------------------------------------------
def phase_bowling_chart(
    phase_df: pd.DataFrame,
    player_name: str = "",
) -> matplotlib.figure.Figure:
    """
    Grouped bar chart: Economy and Dot-ball % per phase.

    Input: output of analytics.bowling.get_phase_bowling_stats()
    Columns needed: phase, phase_label, economy, dot_pct

    WHY ECONOMY + DOT %: Economy alone misses pressure bowling.
    A bowler with 8.0 economy but 55% dot balls is creating massive pressure;
    a bowler with 7.0 economy but 30% dots is just hitting the pitch.
    """
    if phase_df.empty or "economy" not in phase_df.columns:
        return empty_chart("No phase bowling data available")

    df     = phase_df.copy().fillna(0)
    labels = df["phase_label"].tolist()
    econ   = df["economy"].tolist()
    dots   = df["dot_pct"].tolist()

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = make_figure(width=9, height=5)

    ax.bar(x - width / 2, econ, width, label="Economy",    color=PALETTE["negative"],  alpha=0.85)
    ax.bar(x + width / 2, dots, width, label="Dot Ball %", color=PALETTE["positive"],  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    title = f"{player_name} — Bowling by Phase" if player_name else "Bowling by Phase"
    ax.set_title(title, fontweight="bold")
    bar_labels(ax, fmt="{:.1f}", padding=0.3)
    ax.legend(framealpha=0)

    # Economy reference line at 8.0 (IPL average)
    ax.axhline(y=8.0, linestyle="--", color=PALETTE["neutral"], linewidth=1,
               label="IPL avg economy (8.0)")

    plt.close("all")
    return fig


# ---------------------------------------------------------------------------
# Over-by-over economy — colour-coded bar
# ---------------------------------------------------------------------------
def economy_by_over_chart(
    over_df: pd.DataFrame,
    player_name: str = "",
) -> matplotlib.figure.Figure:
    """
    Bar chart showing economy in each over, coloured green/red relative to
    the IPL average of 8.0 runs/over.

    Input: output of analytics.bowling.get_economy_by_over()
    Columns needed: over_number, economy

    WHY GREEN/RED: Immediately flags which overs are the bowler's strength
    zones (green ≤ 8.0) vs danger zones (red > 8.0).
    """
    if over_df.empty or "economy" not in over_df.columns:
        return empty_chart("No over-level bowling data available")

    df = over_df.copy().fillna(8.0)

    IPL_AVG_ECONOMY = 8.0
    colors = [
        PALETTE["positive"] if e <= IPL_AVG_ECONOMY else PALETTE["negative"]
        for e in df["economy"]
    ]

    fig, ax = make_figure(width=11, height=4.5)
    ax.bar(df["over_number"], df["economy"], color=colors, alpha=0.85)
    ax.axhline(y=IPL_AVG_ECONOMY, linestyle="--", color=PALETTE["neutral"],
               linewidth=1.2, label=f"IPL avg ({IPL_AVG_ECONOMY})")

    ax.set_xlabel("Over")
    ax.set_ylabel("Economy (runs/over)")
    title = f"{player_name} — Economy by Over" if player_name else "Economy by Over"
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(1, 21))
    ax.legend(framealpha=0, fontsize=9)

    plt.close("all")
    return fig


# ---------------------------------------------------------------------------
# Wicket type breakdown — horizontal bar
# ---------------------------------------------------------------------------
def wicket_type_chart(
    wicket_df: pd.DataFrame,
    player_name: str = "",
) -> matplotlib.figure.Figure:
    """
    Horizontal bar chart of dismissal types taken by the bowler.

    Input: output of analytics.bowling.get_wicket_type_breakdown()
    Columns needed: wicket_kind, count, pct
    """
    if wicket_df.empty or "wicket_kind" not in wicket_df.columns:
        return empty_chart("No wicket data available")

    df = wicket_df.copy()

    fig, ax = make_figure(width=8, height=max(3, len(df) * 0.6 + 1))
    bars = ax.barh(df["wicket_kind"], df["count"], color=PALETTE["primary"], alpha=0.85)

    # Percentage labels on right
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{row['pct']:.0f}%",
            va="center", ha="left",
            fontsize=9, color=PALETTE["text"],
        )

    ax.set_xlabel("Wickets")
    title = f"{player_name} — Wicket Types" if player_name else "Wicket Types"
    ax.set_title(title, fontweight="bold")
    ax.invert_yaxis()

    plt.close("all")
    return fig


# ---------------------------------------------------------------------------
# Bowling season trend — dual axis
# ---------------------------------------------------------------------------
def bowling_season_trend_chart(
    season_df: pd.DataFrame,
    player_name: str = "",
) -> matplotlib.figure.Figure:
    """
    Dual-axis chart: wickets per season (bars) + economy trend (line).

    Input: output of analytics.bowling.get_bowling_season_trend()
    Columns needed: season, wickets, economy
    """
    if season_df.empty or "wickets" not in season_df.columns:
        return empty_chart("No season bowling data available")

    df = season_df.copy().fillna(0)
    x  = range(len(df))

    fig, ax1 = make_figure(width=10, height=5)
    ax2 = ax1.twinx()

    ax1.bar(x, df["wickets"], color=PALETTE["primary"], alpha=0.75, label="Wickets", zorder=2)
    ax1.set_ylabel("Wickets", color=PALETTE["primary"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["primary"])

    ax2.plot(x, df["economy"].tolist(), color=PALETTE["negative"], linewidth=2.5,
             marker="o", markersize=6, label="Economy", zorder=3)
    ax2.set_ylabel("Economy", color=PALETTE["negative"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["negative"])
    ax2.spines["right"].set_color(PALETTE["negative"])

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["season"].tolist(), rotation=45, ha="right")
    title = f"{player_name} — Season Bowling Trend" if player_name else "Season Bowling Trend"
    ax1.set_title(title, fontweight="bold")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0)

    plt.close("all")
    return fig


# ---------------------------------------------------------------------------
# Bowling summary comparison bars
# ---------------------------------------------------------------------------
def bowling_summary_bars(
    summary: dict,
    player_name: str = "",
) -> matplotlib.figure.Figure:
    """
    Horizontal bar chart comparing key bowling metrics against IPL benchmarks.

    IPL benchmarks:
        Economy ≈ 8.0, avg ≈ 30.0, SR ≈ 18.0, dot_pct ≈ 38%

    Bars inverted for economy/avg/SR (lower = better) so green always means good.
    """
    # Normalise to "higher = better" by inverting economy, avg, sr
    IPL = {"economy": 8.0, "avg": 30.0, "sr": 18.0, "dot_pct": 38.0}

    economy  = summary.get("economy")  or IPL["economy"]
    avg      = summary.get("avg")      or IPL["avg"]
    sr       = summary.get("sr")       or IPL["sr"]
    dot_pct  = summary.get("dot_pct")  or IPL["dot_pct"]

    labels    = ["Economy (lower=better)", "Average (lower=better)", "Bowling SR (lower=better)", "Dot Ball %"]
    values    = [economy, avg, sr, dot_pct]
    benchmarks = [IPL["economy"], IPL["avg"], IPL["sr"], IPL["dot_pct"]]

    fig, ax = make_figure(width=9, height=4.5)
    y = np.arange(len(labels))
    h = 0.35

    ax.barh(y + h / 2, values,    h, color=PALETTE["primary"], alpha=0.85, label="Player")
    ax.barh(y - h / 2, benchmarks, h, color=PALETTE["neutral"], alpha=0.6,  label="IPL Avg")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    title = f"{player_name} vs IPL Average" if player_name else "Bowling vs IPL Average"
    ax.set_title(title, fontweight="bold")
    ax.legend(framealpha=0)

    plt.close("all")
    return fig
