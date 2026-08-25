"""
CricketIQ — Win Probability & Simulation Page
===============================================
Interactive win probability calculator + Monte Carlo score simulator.

Layout:
  Tab 1: Win Probability — enter current match state → see probability gauge
  Tab 2: Score Simulator — simulate an innings → see score distribution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import app.cloud  # noqa: F401 — Streamlit secrets → env vars bridge (cloud deployment)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import streamlit as st

from app.utils import cached_db_ok
from app.components import require_db, section
from src.visualization.style import PALETTE, apply_theme, make_figure, empty_chart

apply_theme()
require_db(cached_db_ok())

# ── Model availability checks ────────────────────────────────────────────────
from src.ml.win_probability import WinProbabilityModel, MODEL_PATH
from src.simulation.monte_carlo import MonteCarloSimulator
from src.simulation.distributions import DISTRIBUTIONS_PATH

model_ready = WinProbabilityModel.is_trained()
sim_ready   = MonteCarloSimulator.is_ready()

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🎯 Win Probability & Simulation")
st.caption("ML-powered win probability · Monte Carlo score simulation")

if not model_ready or not sim_ready:
    st.warning(
        "⚠️ ML models not yet trained. Run these once:\n"
        "```\n"
        "python scripts/compute_distributions.py\n"
        "python scripts/train_win_model.py\n"
        "```"
    )
    if not model_ready:
        st.error(f"Missing: `{MODEL_PATH}`")
    if not sim_ready:
        st.error(f"Missing: `{DISTRIBUTIONS_PATH}`")
    st.stop()

# Load model + simulator (cached via @st.cache_resource to avoid reloading)
@st.cache_resource
def load_model():
    return WinProbabilityModel.load()

@st.cache_resource
def load_simulator():
    return MonteCarloSimulator.load(seed=42)

model = load_model()
sim   = load_simulator()

st.success("✅ Win probability model + simulator loaded")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_win, tab_sim = st.tabs(["📊 Win Probability", "🎲 Score Simulator"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — WIN PROBABILITY CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_win:
    section("Live Win Probability Calculator", "📊")
    st.markdown(
        "Enter the current match state for a **2nd-innings chase** "
        "and get the probability that the batting team wins."
    )

    col_inputs, col_result = st.columns([2, 3])

    with col_inputs:
        target         = st.number_input("Target (runs to win)",       min_value=1,   max_value=300, value=180, step=1)
        runs_scored    = st.number_input("Runs scored so far",         min_value=0,   max_value=299, value=95,  step=1)
        wickets_fallen = st.number_input("Wickets fallen",             min_value=0,   max_value=9,   value=2,   step=1)
        balls_bowled   = st.number_input("Legal balls bowled so far",  min_value=0,   max_value=119, value=60,  step=1)
        phase          = st.selectbox("Current phase", ["powerplay", "middle", "death"], index=1)

        # Derived
        required_runs  = max(0, target - runs_scored)
        balls_remaining = max(0, 120 - balls_bowled)

        st.divider()
        st.markdown("**Derived:**")
        dc1, dc2 = st.columns(2)
        dc1.metric("Required runs",     required_runs)
        dc2.metric("Balls remaining",   balls_remaining)

        overs_rem = balls_remaining / 6 if balls_remaining > 0 else 0
        rrr = required_runs / overs_rem if overs_rem > 0 else 36.0
        dc1.metric("Required RR",       f"{min(36, rrr):.2f}")
        crr = runs_scored / (balls_bowled / 6) if balls_bowled > 0 else 0
        dc2.metric("Current RR",        f"{crr:.2f}")

    with col_result:
        # Predict
        win_prob = model.predict(
            runs_scored    = int(runs_scored),
            wickets_fallen = int(wickets_fallen),
            balls_remaining= int(balls_remaining),
            required_runs  = int(required_runs),
            phase          = phase,
        )
        lose_prob = 1.0 - win_prob

        # Big gauge
        fig, ax = plt.subplots(figsize=(6, 3.5), subplot_kw={"projection": "polar"})
        fig.patch.set_facecolor(PALETTE["background"])
        ax.set_facecolor(PALETTE["background"])

        theta = np.linspace(0, np.pi, 300)
        ax.plot(np.linspace(np.pi/2, np.pi, 150), [1]*150, color=PALETTE["positive"], linewidth=14, alpha=0.3)
        ax.plot(np.linspace(0, np.pi/2, 150),     [1]*150, color=PALETTE["negative"], linewidth=14, alpha=0.3)

        needle = np.pi/2 + (win_prob - 0.5) * np.pi
        ax.annotate("", xy=(needle, 0.85), xytext=(needle, 0.05),
                    arrowprops={"arrowstyle": "->", "color": PALETTE["primary"], "lw": 3})

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.4)
        ax.axis("off")

        color = PALETTE["positive"] if win_prob >= 0.5 else PALETTE["negative"]
        ax.text(np.pi/2, 0.45, f"{win_prob:.0%}", ha="center", va="center",
                fontsize=22, fontweight="bold", color=color)
        ax.text(np.pi, 1.3, "Batting team wins", ha="left",  va="center",
                fontsize=9, color=PALETTE["positive"])
        ax.text(0,     1.3, "Bowling team wins", ha="right", va="center",
                fontsize=9, color=PALETTE["negative"])
        fig.suptitle("Win Probability", fontsize=12, color=PALETTE["text"])
        plt.close("all")
        st.pyplot(fig, use_container_width=True)

        # Interpretation
        if win_prob >= 0.75:
            verdict = "🟢 Batting team is in strong control"
        elif win_prob >= 0.55:
            verdict = "🟡 Batting team has the slight edge"
        elif win_prob >= 0.45:
            verdict = "⚪ Evenly balanced contest"
        elif win_prob >= 0.25:
            verdict = "🟠 Bowling team has the advantage"
        else:
            verdict = "🔴 Bowling team is in strong control"

        st.markdown(f"**{verdict}**")
        m1, m2 = st.columns(2)
        m1.metric("Batting team wins",  f"{win_prob:.1%}")
        m2.metric("Bowling team wins",  f"{lose_prob:.1%}")

    st.divider()

    # Probability trajectory (sweep across remaining balls)
    section("Win Probability Trajectory", "📈")
    st.caption("How win probability changes as the innings progresses (projected from current state)")

    traj_balls = list(range(0, balls_remaining + 1, max(1, balls_remaining // 20)))
    traj_probs = []
    for b in traj_balls:
        balls_gone = balls_bowled + b
        balls_left = max(0, 120 - balls_gone)
        over_num   = balls_gone // 6 + 1
        ph         = "powerplay" if over_num <= 6 else ("death" if over_num > 15 else "middle")
        p = model.predict(
            runs_scored    = int(runs_scored + b * (crr / 6)),  # rough linear projection
            wickets_fallen = int(wickets_fallen),
            balls_remaining= int(balls_left),
            required_runs  = max(0, int(required_runs - b * (crr / 6))),
            phase          = ph,
        )
        traj_probs.append(p)

    fig2, ax2 = make_figure(width=10, height=3.5)
    ax2.plot(traj_balls, traj_probs, color=PALETTE["primary"], linewidth=2.5, marker="o", markersize=4)
    ax2.axhline(y=0.5, linestyle="--", color=PALETTE["neutral"], linewidth=1, label="50% parity")
    ax2.fill_between(traj_balls, traj_probs, 0.5,
                     where=[p >= 0.5 for p in traj_probs],
                     alpha=0.15, color=PALETTE["positive"])
    ax2.fill_between(traj_balls, traj_probs, 0.5,
                     where=[p < 0.5 for p in traj_probs],
                     alpha=0.15, color=PALETTE["negative"])
    ax2.set_xlabel("Balls into remaining chase")
    ax2.set_ylabel("Win probability (batting team)")
    ax2.set_ylim(0, 1)
    ax2.set_title("Projected Win Probability Trajectory", fontweight="bold")
    ax2.legend(framealpha=0)
    plt.close("all")
    st.pyplot(fig2, use_container_width=True)
    st.caption("Projection assumes current run rate continues — actual trajectory will differ based on batting/bowling quality.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCORE SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    section("Monte Carlo Score Simulator", "🎲")
    st.markdown(
        "Simulate **10,000 IPL innings** using historical IPL ball-outcome "
        "distributions to predict the score distribution."
    )

    sc1, sc2 = st.columns(2)
    with sc1:
        s_balls_bowled   = st.number_input("Balls already bowled",  0, 119, 0,  key="sim_balls")
        s_runs_so_far    = st.number_input("Runs scored so far",    0, 299, 0,  key="sim_runs")
        s_wickets_fallen = st.number_input("Wickets fallen",        0,   9, 0,  key="sim_wkts")
    with sc2:
        s_n_sims  = st.select_slider("Simulations", [1000, 2000, 5000, 10000], value=5000)
        s_target  = st.number_input("Chase target (0 = none)", 0, 300, 0, key="sim_tgt")

    target_val = int(s_target) if s_target > 0 else None

    with st.spinner(f"Running {s_n_sims:,} simulations..."):
        result = sim.simulate(
            n_simulations  = s_n_sims,
            balls_bowled   = int(s_balls_bowled),
            runs_so_far    = int(s_runs_so_far),
            wickets_fallen = int(s_wickets_fallen),
            target         = target_val,
        )

    # Summary stats
    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean score",      f"{result.mean_score:.0f}")
    m2.metric("Median score",    f"{result.median_score:.0f}")
    m3.metric("10th percentile", f"{result.percentile_10:.0f}")
    m4.metric("90th percentile", f"{result.percentile_90:.0f}")
    if target_val:
        p_reach = result.probability_of_reaching(target_val)
        m5.metric(f"P(reach {target_val})", f"{p_reach:.1%}")

    # Score distribution histogram
    dist_df = result.score_distribution_df(bin_width=5)

    fig3, ax3 = make_figure(width=11, height=4.5)

    bar_colors = []
    for label in dist_df["score_range"]:
        lo = int(label.split("-")[0])
        if target_val and lo >= target_val:
            bar_colors.append(PALETTE["positive"])
        elif target_val and lo < target_val:
            bar_colors.append(PALETTE["negative"])
        else:
            bar_colors.append(PALETTE["primary"])

    ax3.bar(range(len(dist_df)), dist_df["probability"] * 100,
            color=bar_colors, alpha=0.85, width=0.9)

    # Label every 4th bar
    tick_step = max(1, len(dist_df) // 15)
    ax3.set_xticks(range(0, len(dist_df), tick_step))
    ax3.set_xticklabels(dist_df["score_range"].iloc[::tick_step].tolist(),
                        rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Probability (%)")
    title = f"Score Distribution ({s_n_sims:,} simulations)"
    if s_balls_bowled > 0:
        title += f" — Partial innings: {s_runs_so_far}/{s_wickets_fallen} after {s_balls_bowled} balls"
    ax3.set_title(title, fontweight="bold")

    if target_val:
        # Find x-position of target bin
        for i, label in enumerate(dist_df["score_range"]):
            lo = int(label.split("-")[0])
            if lo >= target_val:
                ax3.axvline(x=i - 0.5, linestyle="--", color=PALETTE["secondary"],
                            linewidth=2, label=f"Target: {target_val}")
                break
        ax3.legend(framealpha=0)

    from matplotlib.patches import Patch
    if target_val:
        legend_els = [
            Patch(facecolor=PALETTE["positive"], label=f"≥ {target_val} (chasing team wins)"),
            Patch(facecolor=PALETTE["negative"], label=f"< {target_val} (bowling team wins)"),
        ]
        ax3.legend(handles=legend_els, framealpha=0)

    plt.close("all")
    st.pyplot(fig3, use_container_width=True)

    if target_val:
        st.markdown(
            f"🟢 **{result.probability_of_reaching(target_val):.1%}** of simulated innings "
            f"reach or exceed {target_val}. "
            f"🔴 **{1 - result.probability_of_reaching(target_val):.1%}** fall short."
        )
    else:
        st.markdown(
            f"Expected score: **{result.mean_score:.0f}** "
            f"(range: {result.percentile_10:.0f}–{result.percentile_90:.0f} for 80% of innings)"
        )

    with st.expander("📋 Score distribution table"):
        st.dataframe(
            dist_df.rename(columns={
                "score_range": "Score Range",
                "count": "Simulations",
                "probability": "Probability"
            }).assign(Probability=lambda d: (d["Probability"] * 100).round(2).astype(str) + "%"),
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.caption(
    "Win probability model: Logistic Regression + CalibratedClassifierCV · "
    "Trained on 135,014 IPL deliveries · AUC 0.853 · Brier 0.156"
)
