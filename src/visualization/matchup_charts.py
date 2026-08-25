"""
CricketIQ — Matchup Chart Functions
======================================
Head-to-head batter vs bowler visualisations.
All functions: (DataFrame | dict) → matplotlib.Figure
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.visualization.style import (
    PALETTE,
    PHASE_COLORS,
    bar_labels,
    empty_chart,
    make_figure,
    make_figure_grid,
    FONT_SIZE_LG,
    FONT_SIZE_SM,
)


# ---------------------------------------------------------------------------
# Matchup summary card — stat tiles
# ---------------------------------------------------------------------------
def matchup_summary_chart(
    summary: dict,
    batter_name: str = "Batter",
    bowler_name: str = "Bowler",
) -> matplotlib.figure.Figure:
    """
    Visual stats card: key head-to-head numbers laid out as tiles.

    Input: output of analytics.matchup.get_matchup_summary()
    Keys used: balls_faced, runs, dismissals, sr, economy, dot_pct, dominance_index

    WHY TILES INSTEAD OF BARS: The matchup summary has mixed units (runs,
    percentages, a rate) — a single bar axis would be meaningless.  Tiles
    let each metric breathe in its own space with a colour cue for who's winning.
    """
    if not summary or summary.get("balls_faced") is None:
        return empty_chart("Insufficient matchup data (< 1 ball faced)")

    dominance = summary.get("dominance_index") or 0.0
    batter_col = PALETTE["positive"] if dominance > 0 else PALETTE["neutral"]
    bowler_col = PALETTE["negative"] if dominance < 0 else PALETTE["neutral"]

    stats = [
        ("Balls Faced",    summary.get("balls_faced") or 0,       ""),
        ("Runs",           summary.get("runs")         or 0,       ""),
        ("Dismissals",     summary.get("dismissals")   or 0,       ""),
        ("Strike Rate",    summary.get("sr")           or 0.0,     ""),
        ("Economy",        summary.get("economy")      or 0.0,     "runs/over"),
        ("Dot Ball %",     summary.get("dot_pct")      or 0.0,     "%"),
    ]

    fig, axes = make_figure_grid(rows=2, cols=3, width=10, height=4)

    for ax, (label, value, unit) in zip(axes, stats):
        ax.set_facecolor(PALETTE["background"])
        ax.axis("off")
        val_str = f"{value:.1f}{unit}" if isinstance(value, float) else str(value)
        ax.text(0.5, 0.65, val_str, transform=ax.transAxes,
                ha="center", va="center", fontsize=20, fontweight="bold",
                color=PALETTE["primary"])
        ax.text(0.5, 0.25, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=FONT_SIZE_SM,
                color=PALETTE["neutral"])

    # Title with dominance indicator
    dom_label = (
        f"🟢 {batter_name} leads" if dominance > 0.05
        else f"🔴 {bowler_name} leads" if dominance < -0.05
        else "⚪ Even matchup"
    )
    fig.suptitle(
        f"{batter_name}  vs  {bowler_name}\n{dom_label}",
        fontsize=FONT_SIZE_LG, fontweight="bold", color=PALETTE["text"],
    )

    plt.close("all")
    return fig


# ---------------------------------------------------------------------------
# Phase comparison — side-by-side bars
# ---------------------------------------------------------------------------
def matchup_phase_chart(
    phase_df: pd.DataFrame,
    batter_name: str = "Batter",
    bowler_name: str = "Bowler",
) -> matplotlib.figure.Figure:
    """
    Side-by-side bar chart: SR (batter advantage) vs Economy (bowler advantage)
    by phase.

    Input: output of analytics.matchup.get_matchup_by_phase()
    Columns: phase, phase_label, sr, economy
    """
    if phase_df.empty or "sr" not in phase_df.columns:
        return empty_chart("No phase matchup data")

    df     = phase_df.copy().fillna(0)
    labels = df["phase_label"].tolist() if "phase_label" in df.columns else df["phase"].tolist()
    sr     = df["sr"].tolist()
    econ   = [e * 100 / 6 for e in df["economy"].tolist()]  # economy → runs-per-ball × 100 = SR equivalent

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = make_figure(width=9, height=5)
    ax.bar(x - width / 2, sr,   width, label=f"{batter_name} SR",        color=PALETTE["primary"],   alpha=0.85)
    ax.bar(x + width / 2, econ, width, label=f"{bowler_name} effective SR", color=PALETTE["negative"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Strike Rate (normalised)")
    ax.set_title(f"{batter_name} vs {bowler_name} — Phase Breakdown", fontweight="bold")
    ax.axhline(y=100, linestyle="--", color=PALETTE["neutral"], linewidth=1, label="SR = 100 (parity)")
    ax.legend(framealpha=0)
    bar_labels(ax, fmt="{:.0f}", padding=1)

    plt.close("all")
    return fig


# ---------------------------------------------------------------------------
# Recent form — timeline dots
# ---------------------------------------------------------------------------
def matchup_recent_form_chart(
    form_df: pd.DataFrame,
    batter_name: str = "Batter",
    bowler_name: str = "Bowler",
) -> matplotlib.figure.Figure:
    """
    Scatter-style timeline: each meeting is a dot; size = runs, red = dismissed.

    Input: output of analytics.matchup.get_matchup_recent_form()
    Columns: match_date, runs, dismissed, sr
    """
    if form_df.empty or "runs" not in form_df.columns:
        return empty_chart("Insufficient recent matchup data")

    df = form_df.copy()
    df = df.sort_values("match_date", ascending=True)
    x  = range(len(df))

    fig, ax = make_figure(width=8, height=4)

    for i, (_, row) in enumerate(df.iterrows()):
        color  = PALETTE["negative"] if row.get("dismissed") else PALETTE["positive"]
        size   = max(40, min(400, row["runs"] * 15))   # bubble size proportional to runs
        ax.scatter(i, row["runs"], s=size, color=color, alpha=0.8, zorder=3)
        ax.text(i, row["runs"] + 1.5, str(int(row["runs"])),
                ha="center", va="bottom", fontsize=8, color=PALETTE["text"])

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [str(d)[:10] for d in df["match_date"]] if "match_date" in df.columns else [str(i+1) for i in x],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_ylabel("Runs scored")
    ax.set_title(f"{batter_name} vs {bowler_name} — Recent Meetings", fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=PALETTE["positive"], label="Not dismissed"),
        mpatches.Patch(color=PALETTE["negative"], label="Dismissed"),
    ]
    ax.legend(handles=legend_patches, framealpha=0, fontsize=9)

    plt.close("all")
    return fig


# ---------------------------------------------------------------------------
# Dominance gauge — half-circle meter
# ---------------------------------------------------------------------------
def dominance_gauge_chart(
    dominance_index: float,
    batter_name: str = "Batter",
    bowler_name: str = "Bowler",
) -> matplotlib.figure.Figure:
    """
    A half-circle gauge from -1 (full bowler dominance) to +1 (full batter dominance).
    The needle points to the dominance_index value.

    WHY A GAUGE: Single-number visualisations need visual weight.  A gauge is
    immediately readable as "who's winning" even before reading the number.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])

    # Draw coloured arc: green (batter), grey (neutral), red (bowler)
    theta_full  = np.linspace(0, np.pi, 300)
    theta_green = np.linspace(np.pi / 2, np.pi, 150)   # batter half
    theta_red   = np.linspace(0, np.pi / 2, 150)        # bowler half

    ax.plot(theta_green, [1] * 150, color=PALETTE["positive"], linewidth=12, alpha=0.3)
    ax.plot(theta_red,   [1] * 150, color=PALETTE["negative"], linewidth=12, alpha=0.3)
    ax.plot([np.pi / 2], [1], "o", color=PALETTE["neutral"], markersize=8)  # centre mark

    # Needle: dominance_index ∈ [-1, 1] → angle ∈ [0, π]
    clamped = max(-1.0, min(1.0, dominance_index))
    needle_angle = np.pi / 2 + clamped * np.pi / 2  # -1→0, 0→π/2, +1→π
    ax.annotate(
        "",
        xy=(needle_angle, 0.85),
        xytext=(needle_angle, 0),
        arrowprops={"arrowstyle": "->", "color": PALETTE["primary"], "lw": 2.5},
    )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.3)
    ax.axis("off")

    # Labels
    ax.text(np.pi,     1.25, batter_name, ha="left",  va="center", fontsize=9, color=PALETTE["positive"])
    ax.text(0,         1.25, bowler_name, ha="right", va="center", fontsize=9, color=PALETTE["negative"])
    ax.text(np.pi / 2, 0.4, f"{dominance_index:+.2f}", ha="center", va="center",
            fontsize=13, fontweight="bold", color=PALETTE["primary"])
    fig.suptitle("Dominance Index", fontsize=11, color=PALETTE["text"])

    plt.close("all")
    return fig
