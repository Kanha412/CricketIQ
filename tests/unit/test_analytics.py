"""
Tests for the CricketIQ analytics engine.

All tests are pure: they build a minimal DataFrame, call an analytics
function, and assert on the result — no database, no Streamlit, no mocking.

WHY THIS APPROACH:
    Analytics functions are pure transformations.  By building known-value
    DataFrames we can assert exact numbers, catching regressions immediately.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.metrics import (
    batting_average,
    bowling_average,
    bowling_strike_rate,
    boundary_percentage,
    dot_ball_percentage,
    economy_rate,
    net_run_rate,
    overs_from_deliveries,
    required_run_rate,
    run_rate,
    strike_rate,
)
from src.analytics.batting import (
    get_batting_summary,
    get_dismissal_breakdown,
    get_phase_batting_stats,
    get_batting_season_trend,
)
from src.analytics.bowling import (
    get_bowling_summary,
    get_phase_bowling_stats,
    get_wicket_type_breakdown,
)
from src.analytics.matchup import (
    get_matchup_summary,
    get_matchup_by_phase,
    _dominance_index,
)
from src.analytics.team import (
    get_team_win_rates,
    get_head_to_head,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def _make_delivery(
    match_id: int = 1,
    batter_id: int = 10,
    bowler_id: int = 20,
    batter_name: str = "Batter A",
    bowler_name: str = "Bowler X",
    player_out_id: int | None = None,
    over_number: int = 1,
    ball_in_over: int = 1,
    runs_batter: int = 0,
    runs_extras: int = 0,
    runs_total: int = 0,
    is_legal_delivery: bool = True,
    is_boundary: bool = False,
    is_six: bool = False,
    is_dot_ball: bool = True,
    is_wicket: bool = False,
    wicket_kind: str | None = None,
    phase: str = "powerplay",
    season: str = "2023",
    venue_name: str = "Wankhede",
    innings_id: int = 1,
    match_date: str = "2023-04-01",
) -> dict:
    return {
        "match_id": match_id, "innings_id": innings_id,
        "batter_id": batter_id, "bowler_id": bowler_id,
        "batter_name": batter_name, "bowler_name": bowler_name,
        "player_out_id": player_out_id,
        "over_number": over_number, "ball_in_over": ball_in_over,
        "runs_batter": runs_batter, "runs_extras": runs_extras, "runs_total": runs_total,
        "is_legal_delivery": is_legal_delivery,
        "is_boundary": is_boundary, "is_six": is_six,
        "is_dot_ball": is_dot_ball, "is_wicket": is_wicket,
        "wicket_kind": wicket_kind, "phase": phase,
        "season": season, "venue_name": venue_name, "match_date": match_date,
    }


# ===========================================================================
# metrics.py tests
# ===========================================================================

class TestBattingMetrics:
    def test_batting_average_normal(self):
        assert batting_average(500, 10) == 50.0

    def test_batting_average_zero_dismissals(self):
        assert batting_average(100, 0) is None

    def test_strike_rate_normal(self):
        assert strike_rate(50, 25) == 200.0

    def test_strike_rate_zero_balls(self):
        assert strike_rate(0, 0) is None

    def test_boundary_pct(self):
        assert boundary_percentage(10, 100) == 10.0

    def test_dot_pct(self):
        assert dot_ball_percentage(30, 100) == 30.0

    def test_dot_pct_zero_balls(self):
        assert dot_ball_percentage(0, 0) is None


class TestBowlingMetrics:
    def test_economy_rate_normal(self):
        # 24 runs in 24 legal deliveries = 4 overs → 6.0 economy
        assert economy_rate(24, 24) == 6.0

    def test_economy_rate_zero_deliveries(self):
        assert economy_rate(0, 0) is None

    def test_bowling_average_normal(self):
        assert bowling_average(120, 5) == 24.0

    def test_bowling_average_no_wickets(self):
        assert bowling_average(100, 0) is None

    def test_bowling_strike_rate_normal(self):
        assert bowling_strike_rate(30, 5) == 6.0

    def test_bowling_strike_rate_no_wickets(self):
        assert bowling_strike_rate(30, 0) is None


class TestMatchMetrics:
    def test_run_rate(self):
        assert run_rate(60, 10) == 6.0

    def test_run_rate_zero_overs(self):
        assert run_rate(60, 0) is None

    def test_required_run_rate(self):
        assert required_run_rate(200, 100, 10) == 10.0

    def test_required_run_rate_game_over(self):
        assert required_run_rate(200, 200, 0) is None

    def test_net_run_rate(self):
        assert net_run_rate(200, 20, 150, 20) == 2.5

    def test_overs_from_deliveries(self):
        assert overs_from_deliveries(13) == 2.1   # 2 complete + 1 ball
        assert overs_from_deliveries(12) == 2.0
        assert overs_from_deliveries(7)  == 1.1


# ===========================================================================
# batting.py tests
# ===========================================================================

class TestBattingSummary:
    def test_empty_df_returns_none_values(self):
        result = get_batting_summary(pd.DataFrame())
        assert result["runs"] is None

    def test_basic_stats(self):
        rows = [
            _make_delivery(match_id=1, runs_batter=4, is_legal_delivery=True, is_boundary=True,  is_dot_ball=False),
            _make_delivery(match_id=1, runs_batter=6, is_legal_delivery=True, is_six=True, is_boundary=True, is_dot_ball=False),
            _make_delivery(match_id=1, runs_batter=0, is_legal_delivery=True, is_dot_ball=True),
            _make_delivery(match_id=1, runs_batter=1, is_legal_delivery=True, is_dot_ball=False),
            # Wicket delivery
            _make_delivery(
                match_id=1, runs_batter=0, is_legal_delivery=True, is_dot_ball=True,
                is_wicket=True, wicket_kind="caught", player_out_id=10,
                batter_id=10,
            ),
        ]
        df = pd.DataFrame(rows)
        summary = get_batting_summary(df)

        assert summary["runs"]       == 11
        assert summary["balls_faced"] == 5
        assert summary["fours"]      == 1
        assert summary["sixes"]      == 1
        assert summary["dismissals"] == 1
        assert summary["sr"]         == 220.0
        assert summary["dot_pct"]    == 40.0

    def test_never_dismissed_avg_is_none(self):
        rows = [_make_delivery(match_id=1, runs_batter=50, is_legal_delivery=True, is_dot_ball=False)]
        df = pd.DataFrame(rows)
        summary = get_batting_summary(df)
        assert summary["avg"] is None   # never dismissed → infinite average


class TestPhaseStats:
    def test_phase_order(self):
        rows = [
            _make_delivery(phase="death",     runs_batter=6, is_legal_delivery=True, is_dot_ball=False),
            _make_delivery(phase="powerplay", runs_batter=4, is_legal_delivery=True, is_dot_ball=False),
            _make_delivery(phase="middle",    runs_batter=2, is_legal_delivery=True, is_dot_ball=False),
        ]
        df = pd.DataFrame(rows)
        result = get_phase_batting_stats(df)
        # Canonical order must be powerplay → middle → death
        assert list(result["phase"]) == ["powerplay", "middle", "death"]

    def test_powerplay_sr(self):
        rows = [
            _make_delivery(phase="powerplay", runs_batter=100, is_legal_delivery=True, is_dot_ball=False),
            _make_delivery(phase="powerplay", runs_batter=0,   is_legal_delivery=True, is_dot_ball=True),
        ]
        df = pd.DataFrame(rows)
        result = get_phase_batting_stats(df)
        pp_row = result[result["phase"] == "powerplay"].iloc[0]
        assert pp_row["sr"] == 5000.0   # 100 runs off 2 balls


class TestDismissalBreakdown:
    def test_breakdown_sorted_descending(self):
        rows = [
            _make_delivery(is_wicket=True, wicket_kind="caught",  player_out_id=10, batter_id=10),
            _make_delivery(is_wicket=True, wicket_kind="caught",  player_out_id=10, batter_id=10),
            _make_delivery(is_wicket=True, wicket_kind="bowled",  player_out_id=10, batter_id=10),
        ]
        df = pd.DataFrame(rows)
        result = get_dismissal_breakdown(df)
        assert result.iloc[0]["wicket_kind"] == "caught"
        assert result.iloc[0]["count"] == 2
        assert result.iloc[0]["pct"] == pytest.approx(66.7, abs=0.1)


class TestSeasonTrend:
    def test_multiple_seasons(self):
        rows = [
            _make_delivery(match_id=1, season="2022", runs_batter=30, is_legal_delivery=True, is_dot_ball=False),
            _make_delivery(match_id=2, season="2023", runs_batter=50, is_legal_delivery=True, is_dot_ball=False),
        ]
        df = pd.DataFrame(rows)
        result = get_batting_season_trend(df)
        assert len(result) == 2
        assert list(result["season"]) == ["2022", "2023"]


# ===========================================================================
# bowling.py tests
# ===========================================================================

class TestBowlingSummary:
    def test_empty_df(self):
        result = get_bowling_summary(pd.DataFrame())
        assert result["wickets"] is None

    def test_economy_calculation(self):
        # 6 runs in 6 legal deliveries = 1 over → economy 6.0
        rows = [
            _make_delivery(runs_total=1, is_legal_delivery=True) for _ in range(6)
        ]
        df = pd.DataFrame(rows)
        summary = get_bowling_summary(df)
        assert summary["economy"] == 6.0
        assert summary["runs_conceded"] == 6

    def test_best_bowling(self):
        rows = [
            _make_delivery(match_id=1, runs_total=20, is_legal_delivery=True, is_wicket=True),
            _make_delivery(match_id=1, runs_total=20, is_legal_delivery=True, is_wicket=True),
            _make_delivery(match_id=2, runs_total=30, is_legal_delivery=True, is_wicket=False),
        ]
        df = pd.DataFrame(rows)
        summary = get_bowling_summary(df)
        # Match 1 has 2 wickets, match 2 has 0 → best = 2/40
        assert summary["best_bowling"] == "2/40"


class TestWicketTypeBreakdown:
    def test_returns_correct_types(self):
        rows = [
            _make_delivery(is_wicket=True, wicket_kind="bowled"),
            _make_delivery(is_wicket=True, wicket_kind="bowled"),
            _make_delivery(is_wicket=True, wicket_kind="lbw"),
        ]
        df = pd.DataFrame(rows)
        result = get_wicket_type_breakdown(df)
        assert result.iloc[0]["wicket_kind"] == "bowled"
        assert result.iloc[0]["count"] == 2


# ===========================================================================
# matchup.py tests
# ===========================================================================

class TestMatchupSummary:
    def test_empty_returns_nones(self):
        result = get_matchup_summary(pd.DataFrame())
        assert result["runs"] is None

    def test_dominance_index_batter_winning(self):
        # 50 runs, 10 balls, 0 wickets → batter dominating
        idx = _dominance_index(runs=50, balls=10, dismissals=0)
        assert idx > 0  # positive → batter advantage

    def test_dominance_index_bowler_winning(self):
        # 5 runs, 10 balls, 3 wickets → bowler dominating
        idx = _dominance_index(runs=5, balls=10, dismissals=3)
        assert idx < 0  # negative → bowler advantage

    def test_dominance_zero_balls(self):
        assert _dominance_index(0, 0, 0) is None

    def test_full_matchup(self):
        rows = [
            _make_delivery(runs_batter=4, runs_total=4, is_legal_delivery=True, is_boundary=True,  is_dot_ball=False, phase="powerplay"),
            _make_delivery(runs_batter=0, runs_total=0, is_legal_delivery=True, is_dot_ball=True,  phase="powerplay"),
            _make_delivery(
                runs_batter=0, runs_total=0, is_legal_delivery=True, is_dot_ball=True,
                is_wicket=True, wicket_kind="bowled", player_out_id=10, batter_id=10, phase="middle",
            ),
        ]
        df = pd.DataFrame(rows)
        summary = get_matchup_summary(df)
        assert summary["runs"] == 4
        assert summary["balls_faced"] == 3
        assert summary["dismissals"] == 1


class TestMatchupByPhase:
    def test_phase_split(self):
        rows = [
            _make_delivery(runs_batter=6, runs_total=6, is_legal_delivery=True, is_dot_ball=False, phase="powerplay"),
            _make_delivery(runs_batter=0, runs_total=0, is_legal_delivery=True, is_dot_ball=True,  phase="death"),
        ]
        df = pd.DataFrame(rows)
        result = get_matchup_by_phase(df)
        pp = result[result["phase"] == "powerplay"].iloc[0]
        de = result[result["phase"] == "death"].iloc[0]
        assert pp["sr"] == 600.0
        assert de["sr"] == 0.0


# ===========================================================================
# team.py tests
# ===========================================================================

class TestTeamWinRates:
    def _make_matches(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"team1_name": "MI",  "team2_name": "CSK", "winner_name": "MI"},
            {"team1_name": "MI",  "team2_name": "CSK", "winner_name": "CSK"},
            {"team1_name": "RCB", "team2_name": "MI",  "winner_name": "MI"},
        ])

    def test_win_rates(self):
        df = self._make_matches()
        result = get_team_win_rates(df)
        mi_row = result[result["team"] == "MI"].iloc[0]
        assert mi_row["played"] == 3
        assert mi_row["won"]    == 2
        assert mi_row["win_pct"] == pytest.approx(66.7, abs=0.1)

    def test_sorted_by_win_pct(self):
        df = self._make_matches()
        result = get_team_win_rates(df)
        assert result.iloc[0]["win_pct"] >= result.iloc[1]["win_pct"]


class TestHeadToHead:
    def test_h2h_record(self):
        matches = pd.DataFrame([
            {"team1_name": "MI", "team2_name": "CSK", "winner_name": "MI"},
            {"team1_name": "CSK","team2_name": "MI",  "winner_name": "MI"},
            {"team1_name": "MI", "team2_name": "CSK", "winner_name": "CSK"},
        ])
        result = get_head_to_head(matches, "MI", "CSK")
        assert result["total_matches"] == 3
        assert result["team_a_wins"]   == 2
        assert result["team_b_wins"]   == 1

    def test_no_h2h_data(self):
        matches = pd.DataFrame([
            {"team1_name": "MI", "team2_name": "RCB", "winner_name": "MI"},
        ])
        result = get_head_to_head(matches, "CSK", "KKR")
        assert result["total_matches"] == 0
