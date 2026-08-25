"""
CricketIQ — Reusable Streamlit UI Components
==============================================
Small building blocks used across multiple pages.

WHY A SEPARATE components.py:
    Pages shouldn't contain raw st.metric / CSS / HTML formatting logic.
    Moving it here keeps page files focused on data flow and layout,
    not presentation details.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Styled stat card (metric with background colour)
# ---------------------------------------------------------------------------
def stat_card(label: str, value, delta=None, help_text: str | None = None) -> None:
    """
    Wrapper around st.metric that adds an optional tooltip.
    value can be str, int, or float — auto-formatted.
    """
    if value is None:
        display = "—"
    elif isinstance(value, float):
        display = f"{value:,.2f}"
    elif isinstance(value, int):
        display = f"{value:,}"
    else:
        display = str(value)

    st.metric(label=label, value=display, delta=delta, help=help_text)


# ---------------------------------------------------------------------------
# DB connection guard — call at top of every page
# ---------------------------------------------------------------------------
def require_db(db_ok: bool) -> None:
    """Stop the page with a clear error if DB is not reachable."""
    if not db_ok:
        st.error(
            "❌ **Database unreachable.** "
            "Make sure PostgreSQL is running and `.env` has the correct `DATABASE_URL`."
        )
        st.stop()


# ---------------------------------------------------------------------------
# Season filter sidebar widget
# ---------------------------------------------------------------------------
def season_filter(seasons_df, label: str = "Season") -> int | None:
    """
    Sidebar selectbox for season.  Returns the season_id (int) or None for 'All'.
    """
    if seasons_df.empty:
        return None

    options = ["All seasons"] + seasons_df["season"].tolist()
    choice  = st.sidebar.selectbox(label, options)

    if choice == "All seasons":
        return None

    row = seasons_df[seasons_df["season"] == choice]
    return int(row["id"].iloc[0]) if not row.empty else None


# ---------------------------------------------------------------------------
# Player search box
# ---------------------------------------------------------------------------
def player_search_box(
    players_df,
    label: str = "Search player",
    key: str = "player_search",
    default_name: str | None = None,
) -> tuple[int | None, str | None]:
    """
    Selectbox that returns (player_id, player_name).
    Returns (None, None) if no players available.
    """
    if players_df.empty:
        st.warning("No players found in the database.")
        return None, None

    names = players_df["name"].tolist()
    default_idx = 0
    if default_name and default_name in names:
        default_idx = names.index(default_name)

    selected = st.selectbox(label, names, index=default_idx, key=key)
    if selected is None:
        return None, None

    row = players_df[players_df["name"] == selected]
    pid = int(row["id"].iloc[0]) if not row.empty else None
    return pid, selected


# ---------------------------------------------------------------------------
# Empty state placeholder
# ---------------------------------------------------------------------------
def no_data_message(context: str = "the selected filters") -> None:
    """Show a friendly 'no data' message instead of an empty space."""
    st.info(
        f"📭 No data found for {context}. "
        "Try selecting different filters or check that ingestion ran successfully."
    )


# ---------------------------------------------------------------------------
# Section divider with title
# ---------------------------------------------------------------------------
def section(title: str, icon: str = "") -> None:
    """Render a styled section header."""
    prefix = f"{icon} " if icon else ""
    st.markdown(f"### {prefix}{title}")


# ---------------------------------------------------------------------------
# Formatted result string helper
# ---------------------------------------------------------------------------
def format_result(row) -> str:
    """
    Build a human-readable match result string from a matches DataFrame row.
    e.g.  'MI won by 8 wickets'  or  'CSK won by 34 runs (DLS)'
    """
    winner = row.get("winner_name") or row.get("winner") or "N/R"
    if winner == "N/R" or not winner:
        return "No result"

    runs    = row.get("win_margin_runs")
    wickets = row.get("win_margin_wickets")
    dls     = row.get("dls_applied", False)
    suffix  = " (DLS)" if dls else ""

    if wickets:
        return f"{winner} won by {int(wickets)} wicket{'s' if int(wickets) != 1 else ''}{suffix}"
    if runs:
        return f"{winner} won by {int(runs)} run{'s' if int(runs) != 1 else ''}{suffix}"
    return f"{winner} won{suffix}"
