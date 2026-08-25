"""
Tests for the AI agent layer.

These tests verify:
  1. Tool registry is populated correctly
  2. Tool functions return JSON-serialisable dicts
  3. Tool functions handle bad inputs gracefully (error dict, not exceptions)
  4. Agent creation fails cleanly with a placeholder key
  5. is_api_key_configured returns correct values

No real DB or Gemini API calls are made — we mock the DB layer.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ai.tools import TOOL_FUNCTIONS, _resolve_player, _df_to_records, _safe
from src.ai.agent import is_api_key_configured


# ===========================================================================
# Tool registry
# ===========================================================================

class TestToolRegistry:
    def test_all_tools_present(self):
        expected = {
            "get_batting_stats",
            "get_bowling_stats",
            "get_matchup",
            "get_team_head_to_head",
            "predict_win_probability",
            "simulate_innings_score",
            "search_player_name",
            "get_venue_stats",
        }
        assert set(TOOL_FUNCTIONS.keys()) == expected

    def test_all_tools_callable(self):
        for name, fn in TOOL_FUNCTIONS.items():
            assert callable(fn), f"{name} is not callable"


# ===========================================================================
# Helper functions
# ===========================================================================

class TestSafe:
    def test_none_returns_none(self):
        assert _safe(None) is None

    def test_float_formats_correctly(self):
        assert _safe(45.678) == "45.68"

    def test_custom_format(self):
        assert _safe(0.876, "{:.1%}") == "87.6%"

    def test_non_numeric_returns_str(self):
        assert _safe("abc") == "abc"


class TestDfToRecords:
    def test_empty_returns_empty_list(self):
        result = _df_to_records(pd.DataFrame())
        assert result == []

    def test_returns_list_of_dicts(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.14, 2.71]})
        result = _df_to_records(df)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_max_rows_respected(self):
        df = pd.DataFrame({"x": range(100)})
        result = _df_to_records(df, max_rows=5)
        assert len(result) <= 5

    def test_floats_rounded(self):
        df = pd.DataFrame({"val": [3.14159]})
        result = _df_to_records(df)
        assert result[0]["val"] == 3.14

    def test_json_serialisable(self):
        df = pd.DataFrame({"a": [1], "b": [3.14], "c": ["hello"]})
        result = _df_to_records(df)
        # Should not raise
        json.dumps(result)


# ===========================================================================
# _resolve_player (mocked)
# ===========================================================================

class TestResolvePlayer:
    def test_returns_none_when_not_found(self):
        with patch("src.ai.tools.search_players") as mock_search:
            mock_search.return_value = pd.DataFrame()
            pid, name = _resolve_player("Nonexistent Player XYZ")
        assert pid is None
        assert name is None

    def test_returns_first_result(self):
        mock_df = pd.DataFrame({"id": [42], "name": ["Virat Kohli"]})
        with patch("src.ai.tools.search_players") as mock_search:
            mock_search.return_value = mock_df
            pid, name = _resolve_player("Kohli")
        assert pid == 42
        assert name == "Virat Kohli"


# ===========================================================================
# get_batting_stats (mocked DB)
# ===========================================================================

class TestGetBattingStats:
    def test_player_not_found_returns_error(self):
        with patch("src.ai.tools.search_players") as mock_search:
            mock_search.return_value = pd.DataFrame()
            result = TOOL_FUNCTIONS["get_batting_stats"]("Unknown Player")
        assert "error" in result

    def test_returns_dict(self):
        """Mock at the analytics layer so we don't need a full DB-shaped DataFrame."""
        mock_players = pd.DataFrame({"id": [1], "name": ["Virat Kohli"]})
        mock_summary = {
            "matches": 200, "innings": 195, "runs": 6500,
            "avg": 37.2, "sr": 132.1, "fours": 650, "sixes": 210,
            "fifties": 40, "hundreds": 5, "highest_score": 113,
            "dot_pct": 35.2, "boundary_pct": 26.8,
        }
        mock_phase_df = pd.DataFrame({
            "phase_label": ["Powerplay", "Middle", "Death"],
            "balls_faced": [500, 1200, 400],
            "runs": [600, 1500, 500],
            "sr": [120.0, 125.0, 125.0],
            "avg": [35.0, 38.0, 30.0],
            "dot_pct": [32.0, 30.0, 28.0],
        })
        mock_dismiss_df = pd.DataFrame({
            "wicket_kind": ["caught", "bowled", "lbw"],
            "count": [80, 30, 20],
            "pct": [60.0, 22.5, 15.0],
        })

        with patch("src.ai.tools.search_players", return_value=mock_players), \
             patch("src.ai.tools.get_player_deliveries", return_value=pd.DataFrame({"_dummy": [1]})), \
             patch("src.ai.tools.get_batting_summary", return_value=mock_summary), \
             patch("src.ai.tools.get_phase_batting_stats", return_value=mock_phase_df), \
             patch("src.ai.tools.get_batting_season_trend", return_value=pd.DataFrame()), \
             patch("src.ai.tools.get_dismissal_breakdown", return_value=mock_dismiss_df):
            result = TOOL_FUNCTIONS["get_batting_stats"]("Kohli")

        assert isinstance(result, dict)
        assert "error" not in result
        assert result["player_name"] == "Virat Kohli"
        assert result["runs"] == 6500
        # Must be JSON serialisable
        json.dumps(result)

    def test_empty_deliveries_returns_error(self):
        mock_players = pd.DataFrame({"id": [1], "name": ["Virat Kohli"]})
        with patch("src.ai.tools.search_players", return_value=mock_players), \
             patch("src.ai.tools.get_player_deliveries", return_value=pd.DataFrame()):
            result = TOOL_FUNCTIONS["get_batting_stats"]("Kohli")
        assert "error" in result


# ===========================================================================
# get_bowling_stats (mocked DB)
# ===========================================================================

class TestGetBowlingStats:
    def test_player_not_found_returns_error(self):
        with patch("src.ai.tools.search_players", return_value=pd.DataFrame()):
            result = TOOL_FUNCTIONS["get_bowling_stats"]("Unknown Bowler")
        assert "error" in result

    def test_returns_dict_json_safe(self):
        """Mock at the analytics layer so we don't need a full DB-shaped DataFrame."""
        mock_players = pd.DataFrame({"id": [2], "name": ["Jasprit Bumrah"]})
        mock_summary = {
            "matches": 130, "overs": 480.4, "wickets": 145,
            "economy": 7.38, "avg": 22.3, "sr": 18.1,
            "dot_pct": 48.2, "best_bowling": "5/10",
            "four_wicket_hauls": 3, "five_wicket_hauls": 1,
        }
        mock_phase_df = pd.DataFrame({
            "phase_label": ["Powerplay", "Middle", "Death"],
            "overs": [160.0, 220.0, 100.0],
            "wickets": [40, 65, 40],
            "economy": [6.8, 7.2, 9.1],
            "dot_pct": [52.0, 48.0, 40.0],
        })
        mock_wkt_df = pd.DataFrame({
            "wicket_kind": ["bowled", "caught", "lbw"],
            "count": [50, 70, 25],
            "pct": [34.5, 48.3, 17.2],
        })

        with patch("src.ai.tools.search_players", return_value=mock_players), \
             patch("src.ai.tools.get_player_deliveries", return_value=pd.DataFrame({"_dummy": [1]})), \
             patch("src.ai.tools.get_bowling_summary", return_value=mock_summary), \
             patch("src.ai.tools.get_phase_bowling_stats", return_value=mock_phase_df), \
             patch("src.ai.tools.get_wicket_type_breakdown", return_value=mock_wkt_df):
            result = TOOL_FUNCTIONS["get_bowling_stats"]("Bumrah")

        assert isinstance(result, dict)
        assert "error" not in result
        assert result["wickets"] == 145
        json.dumps(result)


# ===========================================================================
# get_matchup (mocked)
# ===========================================================================

class TestGetMatchup:
    def test_same_player_id_returns_error(self):
        mock_players = pd.DataFrame({"id": [1], "name": ["Player A"]})
        with patch("src.ai.tools.search_players", return_value=mock_players):
            result = TOOL_FUNCTIONS["get_matchup"]("Player A", "Player A")
        assert "error" in result

    def test_batter_not_found_returns_error(self):
        with patch("src.ai.tools.search_players", return_value=pd.DataFrame()):
            result = TOOL_FUNCTIONS["get_matchup"]("Unknown", "Bumrah")
        assert "error" in result

    def test_empty_matchup_returns_error_key(self):
        mock_batter = pd.DataFrame({"id": [1], "name": ["Kohli"]})
        mock_bowler = pd.DataFrame({"id": [2], "name": ["Bumrah"]})

        def mock_search(query):
            if "Kohli" in query or "kohli" in query.lower():
                return mock_batter
            return mock_bowler

        with patch("src.ai.tools.search_players", side_effect=mock_search), \
             patch("src.ai.tools.get_batter_bowler_deliveries", return_value=pd.DataFrame()):
            result = TOOL_FUNCTIONS["get_matchup"]("Kohli", "Bumrah")

        # Empty matchup should produce an error message (players never met)
        assert "error" in result or "batter" in result


# ===========================================================================
# search_player_name (mocked)
# ===========================================================================

class TestSearchPlayerName:
    def test_returns_matches_list(self):
        mock_df = pd.DataFrame({
            "id": [1, 2],
            "name": ["Virat Kohli", "Virat Singh"],
        })
        with patch("src.ai.tools.search_players", return_value=mock_df):
            result = TOOL_FUNCTIONS["search_player_name"]("Virat")
        assert "matches" in result
        assert isinstance(result["matches"], list)
        assert len(result["matches"]) == 2

    def test_no_results_returns_empty_matches(self):
        with patch("src.ai.tools.search_players", return_value=pd.DataFrame()):
            result = TOOL_FUNCTIONS["search_player_name"]("xyzabc")
        assert result["matches"] == []
        assert "message" in result

    def test_result_json_safe(self):
        mock_df = pd.DataFrame({"id": [1], "name": ["Test Player"]})
        with patch("src.ai.tools.search_players", return_value=mock_df):
            result = TOOL_FUNCTIONS["search_player_name"]("Test")
        json.dumps(result)


# ===========================================================================
# get_team_head_to_head (mocked)
# ===========================================================================

class TestGetTeamHeadToHead:
    def test_no_matches_returns_error(self):
        empty_matches = pd.DataFrame(columns=["team1", "team2", "winner", "venue", "season"])
        with patch("src.ai.tools.get_matches", return_value=empty_matches):
            result = TOOL_FUNCTIONS["get_team_head_to_head"]("Team A", "Team B")
        assert "error" in result

    def test_returns_win_counts(self):
        mock_matches = pd.DataFrame({
            "team1":  ["Mumbai Indians", "Chennai Super Kings", "Mumbai Indians"],
            "team2":  ["Chennai Super Kings", "Mumbai Indians", "Chennai Super Kings"],
            "winner": ["Mumbai Indians", "Mumbai Indians", "Chennai Super Kings"],
            "venue":  ["Wankhede", "Chepauk", "Eden Gardens"],
            "season": ["2023/24", "2022", "2021"],
        })
        with patch("src.ai.tools.get_matches", return_value=mock_matches):
            result = TOOL_FUNCTIONS["get_team_head_to_head"]("Mumbai Indians", "Chennai Super Kings")
        assert "error" not in result
        assert result["total_matches"] == 3
        json.dumps(result)


# ===========================================================================
# is_api_key_configured
# ===========================================================================

class TestIsApiKeyConfigured:
    def test_placeholder_key_returns_false(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key-placeholder"}):
            assert is_api_key_configured() is False

    def test_empty_key_returns_false(self):
        env = {k: v for k, v in os.environ.items() if k != "LLM_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            assert is_api_key_configured() is False

    def test_real_key_returns_true(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "AIzaRealKey12345"}):
            assert is_api_key_configured() is True


# ===========================================================================
# CricketIQAgent.create() fails cleanly with placeholder key
# ===========================================================================

class TestAgentCreate:
    def test_placeholder_key_raises_runtime_error(self):
        from src.ai.agent import CricketIQAgent
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key-placeholder"}):
            with pytest.raises(RuntimeError, match="placeholder_key"):
                CricketIQAgent.create()

    def test_missing_key_raises_runtime_error(self):
        from src.ai.agent import CricketIQAgent
        env = {k: v for k, v in os.environ.items() if k != "LLM_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError):
                CricketIQAgent.create()
