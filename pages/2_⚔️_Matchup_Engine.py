"""
CricketIQ — Matchup Engine Page
==================================
Head-to-head analysis between any batter and bowler.

Layout:
  Sidebar  — batter selector, bowler selector
  Main     — dominance gauge → 6-tile summary card
           → phase comparison chart → recent form timeline
           → raw data expander
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import app.cloud  # noqa: F401 — Streamlit secrets → env vars bridge (cloud deployment)

import streamlit as st

from app.utils import (
    cached_db_ok,
    cached_matchup_deliveries,
    cached_players,
    build_name_id_map,
)
from app.components import require_db, no_data_message, section

from src.analytics.matchup import (
    get_matchup_summary,
    get_matchup_by_phase,
    get_matchup_recent_form,
)
from src.visualization.matchup_charts import (
    dominance_gauge_chart,
    matchup_summary_chart,
    matchup_phase_chart,
    matchup_recent_form_chart,
)
from src.visualization.style import apply_theme

apply_theme()
require_db(cached_db_ok())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚔️ Matchup Engine")
    st.divider()

    players_df = cached_players()
    if players_df.empty:
        st.error("No players in database.")
        st.stop()

    name_to_id = build_name_id_map(players_df)
    player_names = players_df["name"].tolist()

    # Batter selector
    st.markdown("**🏏 Batter**")
    default_batter = player_names.index("V Kohli") if "V Kohli" in player_names else 0
    batter_name = st.selectbox("Select batter", player_names, index=default_batter, key="batter_sel")
    batter_id   = name_to_id[batter_name]

    st.markdown("**⚾ Bowler**")
    default_bowler = player_names.index("JJ Bumrah") if "JJ Bumrah" in player_names else 1
    bowler_name = st.selectbox("Select bowler", player_names, index=default_bowler, key="bowler_sel")
    bowler_id   = name_to_id[bowler_name]

    st.divider()
    st.caption("Data: IPL 2008–2026 · Cricsheet.org")

# ── Guard: same player ────────────────────────────────────────────────────────
if batter_id == bowler_id:
    st.warning("⚠️ Please select a different batter and bowler.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚔️ Matchup Engine")
st.subheader(f"{batter_name}  vs  {bowler_name}")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Fetching head-to-head deliveries..."):
    df = cached_matchup_deliveries(batter_id, bowler_id)

if df.empty:
    no_data_message(f"{batter_name} vs {bowler_name} (they may never have met in IPL)")
    st.stop()

balls  = int(df["is_legal_delivery"].sum())
matches = df["match_id"].nunique()
st.success(f"**{balls}** balls faced across **{matches}** matches")
st.divider()

# ── Dominance gauge ────────────────────────────────────────────────────────────
summary  = get_matchup_summary(df)
dom_idx  = summary.get("dominance_index") or 0.0

section("Dominance Index", "🎯")
col_gauge, col_explain = st.columns([2, 3])
with col_gauge:
    st.pyplot(dominance_gauge_chart(dom_idx, batter_name, bowler_name), use_container_width=True)
with col_explain:
    st.markdown(f"""
    **How to read the gauge:**

    The dominance index combines the batter's strike rate deviation from 100
    and the bowler's wicket frequency into a single number:

    > `(SR − 100) / 100 − wickets_per_ball`

    | Range | Interpretation |
    |---|---|
    | > +0.2 | **Batter dominant** — gets on top regularly |
    | −0.2 to +0.2 | **Even contest** |
    | < −0.2 | **Bowler dominant** — consistently restricts/dismisses |

    **Current:** `{dom_idx:+.3f}` → {'🟢 ' + batter_name + ' has the edge' if dom_idx > 0.05 else '🔴 ' + bowler_name + ' has the edge' if dom_idx < -0.05 else '⚪ Evenly matched'}
    """)

st.divider()

# ── Summary tile card ─────────────────────────────────────────────────────────
section("Head-to-Head Summary", "📊")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Balls Faced",   balls)
m2.metric("Runs",          summary.get("runs") or 0)
m3.metric("Dismissals",    summary.get("dismissals") or 0)
m4.metric("Strike Rate",   f"{summary.get('sr') or 0:.1f}" if summary.get("sr") else "—")
m5.metric("Economy",       f"{summary.get('economy') or 0:.2f}" if summary.get("economy") else "—")
m6.metric("Dot Ball %",    f"{summary.get('dot_pct') or 0:.1f}%" if summary.get("dot_pct") else "—")

# Secondary row
m7, m8, m9, _, _, _ = st.columns(6)
m7.metric("Fours",         summary.get("fours") or 0)
m8.metric("Sixes",         summary.get("sixes") or 0)
m9.metric("Boundary %",    f"{summary.get('boundary_pct') or 0:.1f}%" if summary.get("boundary_pct") else "—")

st.divider()

# ── Phase breakdown ────────────────────────────────────────────────────────────
section("Matchup by Phase", "📐")
phase_df = get_matchup_by_phase(df)
st.pyplot(matchup_phase_chart(phase_df, batter_name, bowler_name), use_container_width=True)
st.caption(
    "Bars show normalised SR: batter's actual SR vs bowler's effective SR "
    "(economy × 100/6). Above 100 = batter ahead; below 100 = bowler ahead."
)

with st.expander("📋 Phase breakdown (raw)"):
    if not phase_df.empty:
        show_cols = [c for c in ["phase_label", "balls_faced", "runs", "dismissals",
                                  "sr", "economy", "dot_pct"] if c in phase_df.columns]
        st.dataframe(phase_df[show_cols], use_container_width=True, hide_index=True)

st.divider()

# ── Recent form ───────────────────────────────────────────────────────────────
section("Recent Meetings", "📅")
form_df = get_matchup_recent_form(df, last_n=8)
st.pyplot(matchup_recent_form_chart(form_df, batter_name, bowler_name), use_container_width=True)
st.caption(
    "Each bubble = one match. Size ∝ runs scored. 🟢 Not dismissed · 🔴 Dismissed"
)

with st.expander("📋 All meetings (raw)"):
    show = df.groupby("match_id").agg(
        date=("match_date", "first"),
        balls=("is_legal_delivery", "sum"),
        runs=("runs_batter", "sum"),
        dismissed=("is_wicket", "any"),
    ).reset_index(drop=True)
    show["dismissed"] = show["dismissed"].map({True: "Yes", False: "No"})
    st.dataframe(show, use_container_width=True, hide_index=True)

st.divider()
st.caption("CricketIQ · Data: Cricsheet.org · IPL 2008–2026")
