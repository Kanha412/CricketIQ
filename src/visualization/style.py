"""
CricketIQ — Chart Style & Theme
=================================
Central place for all visual constants so every chart looks cohesive.

WHY A SINGLE STYLE MODULE:
    Changing the color palette or font size in one place propagates to every
    chart automatically.  If we later add dark-mode support, we change only
    this file.

USAGE:
    from src.visualization.style import apply_theme, PALETTE, make_figure
    fig, ax = make_figure()
    apply_theme()            # call once per process (sets rcParams globally)
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure

# ---------------------------------------------------------------------------
# Color palette — IPL-inspired (deep blue, gold, coral, teal, slate)
# ---------------------------------------------------------------------------
PALETTE = {
    "primary":    "#1A3A5C",   # deep navy — dominant bars, lines
    "secondary":  "#F4A300",   # IPL gold — accent, highlights
    "positive":   "#2ECC71",   # green — good stats (high SR, low economy)
    "negative":   "#E74C3C",   # red — pressure stats (wickets, dot balls)
    "neutral":    "#95A5A6",   # grey — background bars, grid
    "powerplay":  "#3498DB",   # blue — powerplay phase
    "middle":     "#9B59B6",   # purple — middle overs
    "death":      "#E67E22",   # orange — death overs
    "background": "#F8F9FA",   # near-white — figure background
    "text":       "#2C3E50",   # dark slate — all labels
    "grid":       "#E0E0E0",   # light grey — gridlines
}

PHASE_COLORS = {
    "powerplay": PALETTE["powerplay"],
    "middle":    PALETTE["middle"],
    "death":     PALETTE["death"],
}

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT_FAMILY  = "DejaVu Sans"   # always available in matplotlib
FONT_SIZE_SM = 9
FONT_SIZE_MD = 11
FONT_SIZE_LG = 13
FONT_SIZE_XL = 15


# ---------------------------------------------------------------------------
# rcParams — call once at startup (Streamlit app __init__ or test conftest)
# ---------------------------------------------------------------------------
def apply_theme() -> None:
    """
    Apply CricketIQ's visual theme to all subsequent matplotlib figures.
    Idempotent — safe to call multiple times.
    """
    matplotlib.rcParams.update({
        # Fonts
        "font.family":       FONT_FAMILY,
        "font.size":         FONT_SIZE_MD,
        "axes.titlesize":    FONT_SIZE_LG,
        "axes.labelsize":    FONT_SIZE_MD,
        "xtick.labelsize":   FONT_SIZE_SM,
        "ytick.labelsize":   FONT_SIZE_SM,
        "legend.fontsize":   FONT_SIZE_SM,
        # Colors
        "figure.facecolor":  PALETTE["background"],
        "axes.facecolor":    PALETTE["background"],
        "axes.edgecolor":    PALETTE["grid"],
        "axes.grid":         True,
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.6,
        "text.color":        PALETTE["text"],
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["text"],
        "ytick.color":       PALETTE["text"],
        # Layout
        "figure.autolayout": True,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Figure factory
# ---------------------------------------------------------------------------
def make_figure(
    width: float = 8.0,
    height: float = 4.5,
    tight: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Create a single-axes figure with the CricketIQ theme applied.

    Returns (fig, ax) — callers plot on ax, then return fig.
    Streamlit renders it with st.pyplot(fig).

    WHY NOT plt.subplots() DIRECTLY:
        This factory ensures consistent size and theme every time without
        having to repeat apply_theme() in every chart function.
    """
    apply_theme()
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])
    if tight:
        fig.tight_layout(pad=1.5)
    return fig, ax


def make_figure_grid(
    rows: int,
    cols: int,
    width: float = 12.0,
    height: float = 5.0,
) -> tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]:
    """
    Create a multi-axes figure for side-by-side comparisons.
    Returns (fig, [ax1, ax2, ...]) flattened.
    """
    apply_theme()
    fig, axes = plt.subplots(rows, cols, figsize=(width, height))
    fig.patch.set_facecolor(PALETTE["background"])
    flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in flat:
        ax.set_facecolor(PALETTE["background"])
    fig.tight_layout(pad=2.0)
    return fig, list(flat)


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------
def bar_labels(
    ax: matplotlib.axes.Axes,
    fmt: str = "{:.1f}",
    padding: float = 0.5,
    fontsize: int = FONT_SIZE_SM,
) -> None:
    """
    Add value labels on top of every bar in ax.

    WHY: Without labels, tiny bar-height differences are hard to compare
    in a small Streamlit column.  Labels make the chart self-documenting.
    """
    for patch in ax.patches:
        h = patch.get_height()
        if h == 0 or h != h:   # skip zero-height and NaN bars
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            h + padding,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=fontsize,
            color=PALETTE["text"],
        )


def add_subtitle(
    ax: matplotlib.axes.Axes,
    text: str,
    fontsize: int = FONT_SIZE_SM,
) -> None:
    """Add a muted subtitle below the main title."""
    ax.text(
        0.5, 1.02, text,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=fontsize,
        color=PALETTE["neutral"],
    )


def empty_chart(message: str = "No data available") -> matplotlib.figure.Figure:
    """
    Return a blank Figure with a centred message.
    Called by every chart function when the input DataFrame is empty.

    WHY: Crashing on empty data is a terrible user experience.  A clear
    'No data for selected filters' message is much better than a traceback.
    """
    apply_theme()
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])
    ax.text(
        0.5, 0.5, message,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=FONT_SIZE_LG,
        color=PALETTE["neutral"],
    )
    ax.axis("off")
    fig.tight_layout()
    return fig
