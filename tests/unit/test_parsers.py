"""
Unit tests for src/data/parsers.py and src/data/schema_models.py

Tests cover:
    - Standard match parsing
    - Edge cases: wide, no-ball, wicket, super over
    - Phase assignment
    - Derived field computation (is_boundary, is_dot_ball, etc.)
    - Player registry resolution
    - Over indexing (0→1)
    - Season type normalisation (int→str)
"""

import json
from pathlib import Path

import pytest

from src.data.parsers import CricsheetParser, get_phase, parse_match_file
from src.data.schema_models import (
    CricsheetDelivery,
    CricsheetExtras,
    CricsheetMatch,
    CricsheetRuns,
    CricsheetWicket,
)
from src.data.validators import (
    ValidationSeverity,
    has_blocking_errors,
    validate_parsed_match,
)

SAMPLES_DIR = Path(__file__).parent.parent.parent / "data" / "samples"


# ── Phase assignment tests ────────────────────────────────────────────────────

class TestGetPhase:
    def test_t20_powerplay(self):
        assert get_phase(1, "T20") == "powerplay"
        assert get_phase(6, "T20") == "powerplay"

    def test_t20_middle(self):
        assert get_phase(7, "T20") == "middle"
        assert get_phase(15, "T20") == "middle"

    def test_t20_death(self):
        assert get_phase(16, "T20") == "death"
        assert get_phase(20, "T20") == "death"

    def test_it20_same_as_t20(self):
        assert get_phase(1, "IT20") == "powerplay"
        assert get_phase(20, "IT20") == "death"

    def test_odi_powerplay(self):
        assert get_phase(1, "ODI") == "powerplay"
        assert get_phase(10, "ODI") == "powerplay"

    def test_odi_death(self):
        assert get_phase(41, "ODI") == "death"
        assert get_phase(50, "ODI") == "death"

    def test_unknown_match_type_falls_back(self):
        # Unknown type should not raise — returns a phase
        result = get_phase(18, "UNKNOWN")
        assert result in ("powerplay", "middle", "death")


# ── Schema model tests ────────────────────────────────────────────────────────

class TestCricsheetRuns:
    def test_standard_runs(self):
        runs = CricsheetRuns(batter=4, extras=0, total=4)
        assert runs.batter == 4
        assert runs.extras == 0
        assert runs.total == 4

    def test_defaults_to_zero(self):
        runs = CricsheetRuns()
        assert runs.batter == 0
        assert runs.extras == 0
        assert runs.total == 0


class TestCricsheetExtras:
    def test_wide_detected(self):
        extras = CricsheetExtras(wides=1)
        assert extras.is_wide is True
        assert extras.extras_type == "wide"

    def test_no_ball_detected(self):
        extras = CricsheetExtras(noballs=1)
        assert extras.is_no_ball is True
        assert extras.extras_type == "no_ball"

    def test_bye_type(self):
        extras = CricsheetExtras(byes=1)
        assert extras.extras_type == "bye"

    def test_legbye_type(self):
        extras = CricsheetExtras(legbyes=1)
        assert extras.extras_type == "leg_bye"

    def test_no_extras_returns_none(self):
        extras = CricsheetExtras()
        assert extras.extras_type is None
        assert extras.is_wide is False
        assert extras.is_no_ball is False

    def test_wide_takes_priority_over_bye(self):
        # Unusual but theoretically possible
        extras = CricsheetExtras(wides=1, byes=1)
        assert extras.extras_type == "wide"


class TestCricsheetMatchIntegerSeason:
    """Older IPL files have season as an integer (2017), not a string."""
    def test_integer_season_accepted(self):
        raw = {
            "meta": {"data_version": "1.1.0", "created": "2017-04-05", "revision": 1},
            "info": {
                "balls_per_over": 6,
                "dates": ["2017-04-05"],
                "gender": "male",
                "match_type": "T20",
                "outcome": {"winner": "Team A", "by": {"wickets": 5}},
                "overs": 20,
                "season": 2017,        # INTEGER — should be accepted
                "teams": ["Team A", "Team B"],
                "toss": {"winner": "Team A", "decision": "field"},
                "venue": "Stadium",
            },
            "innings": [
                {"team": "Team A", "overs": []}    # Minimal innings to pass model validator
            ],
        }
        # Should not raise ValidationError
        match = CricsheetMatch.model_validate(raw)
        assert match.info.season == 2017


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestCricsheetParser:
    """Tests using actual sample match files."""

    @pytest.fixture
    def sample_files(self):
        """Return list of available sample JSON files."""
        files = list(SAMPLES_DIR.glob("*.json"))
        if not files:
            pytest.skip("No sample files in data/samples/ — run download script first")
        return files

    @pytest.fixture
    def first_sample(self, sample_files):
        """Parse the first available sample file."""
        return parse_match_file(sample_files[0])

    def test_match_has_two_teams(self, first_sample):
        assert first_sample.team1 != ""
        assert first_sample.team2 != ""
        assert first_sample.team1 != first_sample.team2

    def test_match_has_two_innings(self, first_sample):
        assert len(first_sample.innings) == 2

    def test_innings_batting_teams_differ(self, first_sample):
        inn1_team = first_sample.innings[0].batting_team
        inn2_team = first_sample.innings[1].batting_team
        assert inn1_team != inn2_team

    def test_season_is_string(self, first_sample):
        """Season should always be a string after parsing, even if JSON had int."""
        assert isinstance(first_sample.season, str)

    def test_over_numbers_are_one_indexed(self, first_sample):
        """Cricsheet JSON has 0-indexed overs — parser must convert to 1-indexed."""
        all_overs = [
            d.over_number
            for inn in first_sample.innings
            for d in inn.deliveries
        ]
        assert min(all_overs) >= 1, "Over numbers must be 1-indexed (>= 1)"
        assert max(all_overs) <= 20, "T20 has max 20 overs"

    def test_phases_assigned_to_all_deliveries(self, first_sample):
        valid_phases = {"powerplay", "middle", "death", "full"}
        for inn in first_sample.innings:
            for d in inn.deliveries:
                assert d.phase in valid_phases, f"Invalid phase: {d.phase}"

    def test_powerplay_overs_correct(self, first_sample):
        """Overs 1-6 should be powerplay in T20."""
        for inn in first_sample.innings:
            for d in inn.deliveries:
                if d.over_number <= 6:
                    assert d.phase == "powerplay", (
                        f"Over {d.over_number} should be powerplay, got {d.phase}"
                    )

    def test_is_legal_delivery_excludes_wides(self, first_sample):
        """Wide deliveries must NOT be legal."""
        for inn in first_sample.innings:
            for d in inn.deliveries:
                if d.extras_type == "wide":
                    assert d.is_legal_delivery is False

    def test_is_legal_delivery_excludes_no_balls(self, first_sample):
        """No-ball deliveries must NOT be legal."""
        for inn in first_sample.innings:
            for d in inn.deliveries:
                if d.extras_type == "no_ball":
                    assert d.is_legal_delivery is False

    def test_is_boundary_correct(self, first_sample):
        """Deliveries with 4 or 6 batter runs should be flagged as boundaries."""
        for inn in first_sample.innings:
            for d in inn.deliveries:
                expected = d.runs_batter in (4, 6)
                assert d.is_boundary == expected

    def test_is_six_correct(self, first_sample):
        """Only 6-run deliveries should be flagged as sixes."""
        for inn in first_sample.innings:
            for d in inn.deliveries:
                expected = d.runs_batter == 6
                assert d.is_six == expected

    def test_is_dot_ball_requires_legal_and_zero_runs(self, first_sample):
        """
        Dot ball = legal delivery + zero total runs.
        A wide is NOT a dot ball even if 0 extra runs were awarded.
        """
        for inn in first_sample.innings:
            for d in inn.deliveries:
                expected = d.is_legal_delivery and d.runs_total == 0
                assert d.is_dot_ball == expected

    def test_cumulative_runs_monotonically_increasing(self, first_sample):
        """Running score should never decrease."""
        for inn in first_sample.innings:
            prev = 0
            for d in inn.deliveries:
                assert d.cumulative_runs >= prev, (
                    f"Cumulative runs decreased: {prev} → {d.cumulative_runs} "
                    f"at over {d.over_number}.{d.ball_in_over}"
                )
                prev = d.cumulative_runs

    def test_cumulative_runs_final_equals_innings_total(self, first_sample):
        """Last delivery's cumulative runs should equal innings total."""
        for inn in first_sample.innings:
            if inn.deliveries:
                last = inn.deliveries[-1]
                assert last.cumulative_runs == inn.total_runs

    def test_wickets_do_not_exceed_ten(self, first_sample):
        """No innings should have more than 10 wickets."""
        for inn in first_sample.innings:
            assert inn.total_wickets <= 10

    def test_player_ids_populated(self, first_sample):
        """Player IDs should be populated from the registry."""
        registry = first_sample.player_registry
        for inn in first_sample.innings:
            for d in inn.deliveries[:10]:    # check first 10 deliveries
                assert d.batter_id != "", "batter_id should not be empty"
                assert d.bowler_id != "", "bowler_id should not be empty"

    def test_all_sample_files_parse_without_errors(self, sample_files):
        """All committed sample files should parse without validation errors."""
        for f in sample_files:
            parsed = parse_match_file(f)
            issues = validate_parsed_match(parsed)
            errors = [i for i in issues if i.is_error]
            assert not errors, f"{f.name} has validation errors: {errors}"


# ── Validator tests ───────────────────────────────────────────────────────────

class TestValidators:
    def test_valid_match_has_no_errors(self, all_sample_matches):
        """Sample matches should have no blocking validation errors."""
        for raw in all_sample_matches:
            match = CricsheetMatch.model_validate(raw)
            parser = CricsheetParser()
            parsed = parser.parse(match, file_id="test")
            issues = validate_parsed_match(parsed)
            assert not has_blocking_errors(issues)

    def test_has_blocking_errors_true_for_errors(self):
        from src.data.validators import ValidationIssue, ValidationSeverity
        issues = [ValidationIssue(ValidationSeverity.ERROR, "field", "msg")]
        assert has_blocking_errors(issues) is True

    def test_has_blocking_errors_false_for_warnings_only(self):
        from src.data.validators import ValidationIssue, ValidationSeverity
        issues = [ValidationIssue(ValidationSeverity.WARNING, "field", "msg")]
        assert has_blocking_errors(issues) is False

    def test_has_blocking_errors_false_for_empty(self):
        assert has_blocking_errors([]) is False
