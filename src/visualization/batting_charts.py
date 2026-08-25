"""
CricketIQ — Batting Chart Functions
======================================
All functions: (DataFrame | dict) → matplotlib.Figure
Callers: Streamlit pages and the AI agent's image-rendering tool.

Inputs come directly from src.analytics.batting functions.
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
    make_figure_grid,
)


# ---------------------------------------------------------------------------
# Phase batting breakdown — bar chart
# ---------------------------------------------------------------------------
def phase_batting_chart(
    phase_df: pd.DataFrame,
    player_name: str = "",
) -> matplotlib.figure.Figure:
    """
    Grouped bar chart: Strike Rate and Average per phase.

    Input: output of analytics.batting.get_phase_batting_stats()
    Columns needed: phase, phase_label, sr, avg

    WHY GROUPED BARS: SR and average tell complementary stories — a player
    can have high SR but low average (aggressive but inconsistent) or the
    opposite.  Side-by-side makes the contrast obvious.
    """
    if phase_df.empty or "sr" not in phase_df.columns:
        return empty_chart("No phase data available")

    df = phase_df.copy().fillna(0)
    labels   = df["phase_label"].tolist()
    sr_vals  = df["sr"].tolist()
    avg_vals = df["avg"].tolist()

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = make_figure(width=9, height=5)

    bars_sr  = ax.bar(x - width / 2, sr_vals,  width, label="Strike Rate", color=PALETTE["primary"],   alpha=0.88)
    bars_avg = ax.bar(x + width / 2, avg_vals, width, label="Average",     color=PALETTE["secondary"], alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Value")
    title = f"{player_name} — Batting by Phase" if player_name else "Batting by Phase"
    ax.set_title(title, fontweight="bold")
    bar_labels(ax, fmt="{:.1f}", padding=0.5)
    ax.legend(framealpha=0)
    ax.set_ylim(0, max(max(sr_vals + [0]), max(avg_vals + [0])) * 1.25)

    plt.close("all")
    return fig


# ---------------------------------------------------------------------------
# Season trend — line chart
# ---------------------------------------------------------------------------
def batting_season_trend_chart(
    season_df: pd.DataFrame,
    player_name: str = "",
) -> matplotlib.figure.Figure:
    """
    Dual-axis line chart: runs per season (bars) + SR overlay (line).

    Input: output of analytics.batting.get_batting_season_trend()
    Columns needed: season, runs, sr

    WHY DUAL AXIS: Runs and SR have different scales but are jointly interesting.
    A single axis would squash the SR line at the bottom.
    """
    if season_df.empty or "runs" not in season_df.columns:
        return empty_chart("No season data available")

    df = season_df.copy().fillna(0)
    x  = range(len(df))

    fig, ax1 = make_figure(width=10, height=5)
    ax2 = ax1.twinx()

    # Runs bars on primary axis
    ax1.bar(x, df["runs"], color=PALETTE["primary"], alpha=0.7, label="Runs", zorder=2)
    ax1.set_ylabel("Total Runs", color=PALETTE["primary"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["primary"])

    # SR line on secondary axis
    sr_vals = df["sr"].tolist()
    ax2.plot(x, sr_vals, color=PALETTE["secondary"], linewidth=2.5,
             marker="o", markersize=6, label="Strike Rate", zorder=3)
    ax2.set_ylabel("Strike Rate", color=PALETTE["secondary"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["secondary"])

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["season"].tolist(), rotation=45, ha="right")
    title = f"{player_name} — Season Batting Trend" if player_name else "Season Batting Trend"
    ax1.set_title(title, fontweight="bold")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0)

    # Remove right spine from ax2 for cleanliness
    ax2.spines["right"].set_color(PALETTE["secondary"])

    plt.close("all")
    return fig


# ---------------------------------------------------------------------------
# Over-by-over scoring map — bar chart
# ---------------------------------------------------------------------------
def scoring_by_over_chart(
    over_df: pd.DataFrame,
    player_name: str = "",
) -> matplotlib.figure.Figure:
    """
    Bar chart coloured by phase showing avg runs per over (1–20).

    Input: output of analytics.batting.get_scoring_by_over()
    Columns needed: over_number, avg_runs_per_over, sr

    WHY PHASE COLOURS: Instantly shows which phase the player is most
    dangerous in without needing a legend explanation.
    """
    if over_df.empty or "over_number" not in over_df.columns:
        return empty_chart("No over-level data available")

    df = over_df.copy().fillna(0)

    def _phase_color(over: int) -> str:
        if over <= 6:
            return PHASE_COLORS["powerplay"]
        if over <= 15:
            return PHASE_COLORS["middle"]
        return PHASE_COLORS["death"]

    colors = [_phase_color(o) for o in df["over_number"]]

    fig, ax = make_figure(width=11, height=4.5)
    ax.bar(df["over_number"], df["avg_runs_per_over"], color=colors, alpha=0.85)

    ax.set_xlabel("Over")
    ax.set_ylabel("Avg Runs Scored (when batting in that over)")
    title = f"{player_name} — Scoring Map by Over" if player_name else "Scoring Map by Over"
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(1, 21))

    # Phase legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PHASE_COLORS["powerplay"], label="Powerplay (1–6)"),
        Patch(facecolor=PHASE_COLORS["middle"],    label="Middle (7–15)"),
        Patch(facecolor=PHASE_COLORS["death"],     label="Death (16–20)"),
    ]
    ax.legend(handles=legend_elements, framealpha=0, fontsize=9)

    plt.close("all")
    return fig


# ---------------------------------------------------------------------------
# Dismissal breakdown — donut chart
# ---------------------------------------------------------------------------
def dismissal_donut_chart(
    dismissal_df: pd.DataFrame,
    player_name: str = "",
) -> matplotlib.figure.Figure:
    """
    Donut chart showing how the player gets out.

    Input: output of analytics.batting.get_dismissal_breakdown()
    Columns needed: wicket_kind, count, pct

    WHY A DONUT: Shows proportions clearly; the hole in the centre lets us
    print the total dismissal count without cluttering the segments.
    """
    if dismissal_df.empty or "wicket_kind" not in dismissal_df.columns:
        return empty_chart("No dismissal data available")

    df = dismissal_df.copy()
    labels = [f"{row['wicket_kind']}\n({int(row['count'])}, {row['pct']:.0f}%)" for _, row in df.iterrows()]
    sizes  = df["count"].tolist()

    # Colour cycle — use as many colours from the palette as needed
    base_colors = [
        PALETTE["primary"], PALETTE["secondary"], PALETTE["negative"],
        PALETTE["positive"], PALETTE["middle"], PALETTE["death"],
    ]
    colors = (base_colors * 10)[:len(sizes)]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])

    wedges, texts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        startangle=140,
        wedgeprops={"width": 0.5, "edgecolor": PALETTE["background"], "linewidth": 2},
    )

    # Centre text: total dismissals
    total = sum(sizes)
    ax.text(0, 0, f"{total}\nout", ha="center", va="center",
            fontsize=14, fontweight="bold", color=PALETTE["text"])

    # Legend outside
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5),
              fontsize=9, framealpha=0)

    title = f"{player_name} — Dismissal Breakdown" if player_name else "Dismissal Breakdown"
    ax.set_title(title, fontweight="bold", pad=15)

    plt.close("all")
    return fig


# ---------------------------------------------------------------------------
# Batting radar / summary card — horizontal bar
# ---------------------------------------------------------------------------
def batting_summary_bars(
    summary: dict,
    player_name: str = "",
) -> matplotlib.figure.Figure:
    """
    Horizontal bar chart comparing key batting metrics against IPL benchmarks.

    Input: output of analytics.batting.get_batting_summary()
    Keys used: sr, avg, boundary_pct, dot_pct

    IPL benchmarks (approximate career averages across all players):
        SR ≈ 135, avg ≈ 28, boundary_pct ≈ 17%, dot_pct ≈ 40%

    WHY BENCHMARKS: A raw SR of 140 means nothing without context.
    Comparing to the league average makes the number immediately interpretable.
    """
    metrics = {
        "Strike Rate":      (summary.get("sr")           or 0, 135.0),
        "Average":          (summary.get("avg")          or 0,  28.0),
        "Boundary %":       (summary.get("boundary_pct") or 0,  17.0),
        "Dot Ball % (inv)": (100 - (summary.get("dot_pct") or 0), 60.0),  # inverted: lower dot = better
    }

    labels    = list(metrics.keys())
    values    = [v[0] for v in metrics.values()]
    benchmarks = [v[1] for v in metrics.values()]

    fig, ax = make_figure(width=8, height=4.5)
    y = np.arange(len(labels))
    height = 0.35

    ax.barh(y + height / 2, values,    height, color=PALETTE["primary"],   alpha=0.85, label="Player")
    ax.barh(y - height / 2, benchmarks, height, color=PALETTE["neutral"],   alpha=0.6,  label="IPL Avg")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Value")
    title = f"{player_name} vs IPL Average" if player_name else "vs IPL Average"
    ax.set_title(title, fontweight="bold")
    ax.legend(framealpha=0)
    ax.invert_yaxis()

    plt.close("all")
    return fig
