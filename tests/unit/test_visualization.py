"""
Visualization smoke tests.

STRATEGY: We don't test pixel-level output — that's brittle and slow.
Instead we verify:
  1. Every function returns a matplotlib Figure (not None, not an error)
  2. Every function handles an empty DataFrame / empty dict gracefully
     (returns a "No data" Figure, never raises)

WHY SMOKE TESTS ONLY:
    The output of a chart function is visual — it can only be truly validated
    by a human looking at it.  What we automate is: "does it run without
    crashing?" which catches import errors, API breakage, and empty-data bugs
    that would otherwise surface live in Streamlit.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required in CI

import matplotlib.figure
import pandas as pd
import pytest

# Style
from src.visualization.style import (
    apply_theme, empty_chart, make_figure, make_figure_grid,
)

# Batting charts
from src.visualization.batting_charts import (
    batting_season_trend_chart,
    batting_summary_bars,
    dismissal_donut_chart,
    phase_batting_chart,
    scoring_by_over_chart,
)

# Bowling charts
from src.visualization.bowling_charts import (
    bowling_season_trend_chart,
    bowling_summary_bars,
    economy_by_over_chart,
    phase_bowling_chart,
    wicket_type_chart,
)

# Matchup charts
from src.visualization.matchup_charts import (
    dominance_gauge_chart,
    matchup_phase_chart,
    matchup_recent_form_chart,
    matchup_summary_chart,
)

# Team charts
from src.visualization.team_charts import (
    head_to_head_chart,
    phase_scoring_chart,
    toss_impact_chart,
    venue_win_rate_chart,
    win_rate_chart,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_figure(obj) -> bool:
    return isinstance(obj, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# style.py
# ---------------------------------------------------------------------------
class TestStyle:
    def test_apply_theme_does_not_crash(self):
        apply_theme()   # idempotent

    def test_make_figure_returns_fig_and_ax(self):
        fig, ax = make_figure()
        assert is_figure(fig)
        assert ax is not None

    def test_make_figure_grid(self):
        fig, axes = make_figure_grid(rows=1, cols=2)
        assert is_figure(fig)
        assert len(axes) == 2

    def test_empty_chart_returns_figure(self):
        fig = empty_chart("test message")
        assert is_figure(fig)

    def test_overs_from_deliveries_covered_elsewhere(self):
        # overs_from_deliveries lives in src.analytics.metrics — covered by test_analytics.py
        pass


# ---------------------------------------------------------------------------
# Batting charts
# ---------------------------------------------------------------------------
class TestBattingCharts:
    def _phase_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"phase": "powerplay", "phase_label": "Powerplay (1–6)", "sr": 142.0, "avg": 35.0, "runs": 280, "balls_faced": 197, "dismissals": 8,  "fours": 24, "sixes": 10, "dot_pct": 35.0, "boundary_pct": 17.0},
            {"phase": "middle",    "phase_label": "Middle (7–15)",   "sr": 128.0, "avg": 28.0, "runs": 350, "balls_faced": 273, "dismissals": 12, "fours": 28, "sixes": 8,  "dot_pct": 42.0, "boundary_pct": 13.0},
            {"phase": "death",     "phase_label": "Death (16–20)",   "sr": 175.0, "avg": 22.0, "runs": 210, "balls_faced": 120, "dismissals": 9,  "fours": 14, "sixes": 12, "dot_pct": 30.0, "boundary_pct": 22.0},
        ])

    def _season_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"season": "2021", "runs": 400, "balls_faced": 290, "dismissals": 12, "avg": 33.3, "sr": 137.9, "fours": 35, "sixes": 18, "boundary_pct": 18.3},
            {"season": "2022", "runs": 520, "balls_faced": 370, "dismissals": 14, "avg": 37.1, "sr": 140.5, "fours": 42, "sixes": 22, "boundary_pct": 17.3},
            {"season": "2023", "runs": 450, "balls_faced": 310, "dismissals": 13, "avg": 34.6, "sr": 145.2, "fours": 38, "sixes": 20, "boundary_pct": 18.7},
        ])

    def _over_df(self) -> pd.DataFrame:
        import numpy as np
        overs = list(range(1, 21))
        return pd.DataFrame({
            "over_number":       overs,
            "balls_faced":       [15] * 20,
            "runs":              [20, 18, 22, 19, 25, 30, 12, 10, 14, 16, 18, 20, 15, 13, 17, 28, 35, 40, 38, 45],
            "avg_runs_per_over": [20/10, 18/10, 22/10, 19/10, 25/10, 30/10, 12/10, 10/10,
                                   14/10, 16/10, 18/10, 20/10, 15/10, 13/10, 17/10, 28/10, 35/10, 40/10, 38/10, 45/10],
            "sr":                [133] * 20,
        })

    def _dismissal_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"wicket_kind": "caught",  "count": 18, "pct": 60.0},
            {"wicket_kind": "bowled",  "count":  8, "pct": 26.7},
            {"wicket_kind": "lbw",     "count":  4, "pct": 13.3},
        ])

    def test_phase_batting_chart_returns_figure(self):
        assert is_figure(phase_batting_chart(self._phase_df(), "Virat Kohli"))

    def test_phase_batting_chart_empty(self):
        assert is_figure(phase_batting_chart(pd.DataFrame()))

    def test_season_trend_chart_returns_figure(self):
        assert is_figure(batting_season_trend_chart(self._season_df(), "Rohit Sharma"))

    def test_season_trend_chart_empty(self):
        assert is_figure(batting_season_trend_chart(pd.DataFrame()))

    def test_scoring_by_over_chart_returns_figure(self):
        assert is_figure(scoring_by_over_chart(self._over_df(), "MS Dhoni"))

    def test_scoring_by_over_chart_empty(self):
        assert is_figure(scoring_by_over_chart(pd.DataFrame()))

    def test_dismissal_donut_chart_returns_figure(self):
        assert is_figure(dismissal_donut_chart(self._dismissal_df(), "Rohit Sharma"))

    def test_dismissal_donut_chart_empty(self):
        assert is_figure(dismissal_donut_chart(pd.DataFrame()))

    def test_batting_summary_bars_returns_figure(self):
        summary = {"sr": 145.0, "avg": 38.5, "boundary_pct": 19.2, "dot_pct": 35.0}
        assert is_figure(batting_summary_bars(summary, "Virat Kohli"))

    def test_batting_summary_bars_empty(self):
        assert is_figure(batting_summary_bars({}))


# ---------------------------------------------------------------------------
# Bowling charts
# ---------------------------------------------------------------------------
class TestBowlingCharts:
    def _phase_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"phase": "powerplay", "phase_label": "Powerplay (1–6)", "economy": 6.5,  "dot_pct": 48.0, "wickets": 12, "overs": 24.0, "avg": 22.0, "runs_conceded": 156},
            {"phase": "middle",    "phase_label": "Middle (7–15)",   "economy": 7.2,  "dot_pct": 42.0, "wickets": 18, "overs": 36.0, "avg": 19.0, "runs_conceded": 342},
            {"phase": "death",     "phase_label": "Death (16–20)",   "economy": 9.8,  "dot_pct": 30.0, "wickets": 10, "overs": 18.0, "avg": 35.0, "runs_conceded": 294},
        ])

    def _over_df(self) -> pd.DataFrame:
        overs = list(range(1, 21))
        return pd.DataFrame({
            "over_number": overs,
            "deliveries":  [10] * 20,
            "runs":        [60, 55, 70, 65, 80, 90, 50, 45, 55, 60, 65, 70, 55, 50, 60, 95, 110, 125, 120, 140],
            "economy":     [6.0, 5.5, 7.0, 6.5, 8.0, 9.0, 5.0, 4.5, 5.5, 6.0, 6.5, 7.0, 5.5, 5.0, 6.0, 9.5, 11.0, 12.5, 12.0, 14.0],
            "dot_pct":     [50.0] * 20,
            "avg_runs_per_over": [6.0] * 20,
        })

    def _wicket_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"wicket_kind": "caught",  "count": 22, "pct": 55.0},
            {"wicket_kind": "bowled",  "count": 10, "pct": 25.0},
            {"wicket_kind": "lbw",     "count":  8, "pct": 20.0},
        ])

    def _season_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"season": "2021", "wickets": 18, "economy": 7.2, "overs": 30.0, "avg": 24.0, "dot_pct": 42.0, "runs_conceded": 432, "matches": 12},
            {"season": "2022", "wickets": 22, "economy": 6.8, "overs": 35.0, "avg": 21.0, "dot_pct": 45.0, "runs_conceded": 476, "matches": 14},
        ])

    def test_phase_bowling_chart_returns_figure(self):
        assert is_figure(phase_bowling_chart(self._phase_df(), "Jasprit Bumrah"))

    def test_phase_bowling_chart_empty(self):
        assert is_figure(phase_bowling_chart(pd.DataFrame()))

    def test_economy_by_over_chart_returns_figure(self):
        assert is_figure(economy_by_over_chart(self._over_df(), "Bumrah"))

    def test_economy_by_over_chart_empty(self):
        assert is_figure(economy_by_over_chart(pd.DataFrame()))

    def test_wicket_type_chart_returns_figure(self):
        assert is_figure(wicket_type_chart(self._wicket_df(), "Bumrah"))

    def test_wicket_type_chart_empty(self):
        assert is_figure(wicket_type_chart(pd.DataFrame()))

    def test_bowling_season_trend_returns_figure(self):
        assert is_figure(bowling_season_trend_chart(self._season_df(), "Bumrah"))

    def test_bowling_season_trend_empty(self):
        assert is_figure(bowling_season_trend_chart(pd.DataFrame()))

    def test_bowling_summary_bars_returns_figure(self):
        summary = {"economy": 6.8, "avg": 21.0, "sr": 14.5, "dot_pct": 46.0}
        assert is_figure(bowling_summary_bars(summary, "Bumrah"))

    def test_bowling_summary_bars_empty(self):
        assert is_figure(bowling_summary_bars({}))


# ---------------------------------------------------------------------------
# Matchup charts
# ---------------------------------------------------------------------------
class TestMatchupCharts:
    def _summary(self) -> dict:
        return {
            "balls_faced": 48, "runs": 60, "dismissals": 3,
            "sr": 125.0, "economy": 7.5, "dot_pct": 37.5,
            "boundary_pct": 20.8, "fours": 7, "sixes": 3,
            "matches_met": 8, "dominance_index": 0.22,
        }

    def _phase_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"phase": "powerplay", "phase_label": "Powerplay (1–6)", "balls_faced": 18, "runs": 25, "dismissals": 1, "sr": 138.9, "economy": 8.3, "dot_pct": 33.3, "boundary_pct": 22.2},
            {"phase": "death",     "phase_label": "Death (16–20)",   "balls_faced": 12, "runs": 10, "dismissals": 2, "sr":  83.3, "economy": 5.0, "dot_pct": 50.0, "boundary_pct": 8.3},
        ])

    def _form_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"match_id": 1, "match_date": "2023-04-15", "balls": 6,  "runs": 10, "dismissed": False, "sr": 166.7},
            {"match_id": 2, "match_date": "2023-05-02", "balls": 8,  "runs": 5,  "dismissed": True,  "sr":  62.5},
            {"match_id": 3, "match_date": "2024-04-20", "balls": 10, "runs": 18, "dismissed": False, "sr": 180.0},
        ])

    def test_summary_chart_returns_figure(self):
        assert is_figure(matchup_summary_chart(self._summary(), "Kohli", "Bumrah"))

    def test_summary_chart_empty(self):
        assert is_figure(matchup_summary_chart({}))

    def test_phase_chart_returns_figure(self):
        assert is_figure(matchup_phase_chart(self._phase_df(), "Kohli", "Bumrah"))

    def test_phase_chart_empty(self):
        assert is_figure(matchup_phase_chart(pd.DataFrame()))

    def test_recent_form_chart_returns_figure(self):
        assert is_figure(matchup_recent_form_chart(self._form_df(), "Kohli", "Bumrah"))

    def test_recent_form_chart_empty(self):
        assert is_figure(matchup_recent_form_chart(pd.DataFrame()))

    def test_dominance_gauge_positive(self):
        assert is_figure(dominance_gauge_chart(0.42, "Kohli", "Bumrah"))

    def test_dominance_gauge_negative(self):
        assert is_figure(dominance_gauge_chart(-0.35, "Kohli", "Bumrah"))

    def test_dominance_gauge_neutral(self):
        assert is_figure(dominance_gauge_chart(0.0))


# ---------------------------------------------------------------------------
# Team charts
# ---------------------------------------------------------------------------
class TestTeamCharts:
    def _win_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"team": "MI",  "played": 220, "won": 130, "win_pct": 59.1},
            {"team": "CSK", "played": 210, "won": 125, "win_pct": 59.5},
            {"team": "RCB", "played": 218, "won": 104, "win_pct": 47.7},
            {"team": "KKR", "played": 215, "won": 105, "win_pct": 48.8},
        ])

    def _venue_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"venue_name": "Wankhede Stadium",      "matches": 80, "bat_first_wins": 38, "bowl_first_wins": 42, "bat_first_win_pct": 47.5},
            {"venue_name": "MA Chidambaram Stadium","matches": 70, "bat_first_wins": 42, "bowl_first_wins": 28, "bat_first_win_pct": 60.0},
            {"venue_name": "M Chinnaswamy Stadium", "matches": 65, "bat_first_wins": 30, "bowl_first_wins": 35, "bat_first_win_pct": 46.2},
        ])

    def _phase_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"phase": "powerplay", "avg_runs": 52.3, "avg_wickets": 1.8, "run_rate": 8.7},
            {"phase": "middle",    "avg_runs": 80.5, "avg_wickets": 3.2, "run_rate": 8.9},
            {"phase": "death",     "avg_runs": 62.1, "avg_wickets": 2.5, "run_rate": 12.4},
        ])

    def test_win_rate_chart_returns_figure(self):
        assert is_figure(win_rate_chart(self._win_df()))

    def test_win_rate_chart_empty(self):
        assert is_figure(win_rate_chart(pd.DataFrame()))

    def test_h2h_chart_returns_figure(self):
        h2h = {"total_matches": 30, "team_a_wins": 17, "team_b_wins": 13, "no_result": 0,
               "team_a_win_pct": 56.7, "team_b_win_pct": 43.3}
        assert is_figure(head_to_head_chart(h2h, "MI", "CSK"))

    def test_h2h_chart_empty(self):
        assert is_figure(head_to_head_chart({}, "MI", "CSK"))

    def test_toss_impact_chart_returns_figure(self):
        data = {"toss_winner_won_match_pct": 52.0, "bat_first_win_pct": 45.0, "field_first_win_pct": 57.0, "total_matches_analysed": 200}
        assert is_figure(toss_impact_chart(data))

    def test_toss_impact_chart_empty(self):
        assert is_figure(toss_impact_chart({}))

    def test_venue_chart_returns_figure(self):
        assert is_figure(venue_win_rate_chart(self._venue_df()))

    def test_venue_chart_empty(self):
        assert is_figure(venue_win_rate_chart(pd.DataFrame()))

    def test_phase_scoring_chart_returns_figure(self):
        assert is_figure(phase_scoring_chart(self._phase_df()))

    def test_phase_scoring_chart_empty(self):
        assert is_figure(phase_scoring_chart(pd.DataFrame()))
