"""
CricketIQ — Data Validators
==============================
Business logic validation for parsed match data.

PHILOSOPHY:
    These validators check the MEANING of the data, not just the structure.
    They return warnings (not exceptions) — invalid data is logged and flagged,
    but ingestion continues where possible. We never want to silently ingest
    corrupted data, but we also don't want to lose entire matches over minor issues.

    Severity levels:
        WARNING  → data quality issue, but match is usable
        ERROR    → match data is fundamentally broken (skip this match)

    The caller (ingestion script) decides what to do based on severity.
"""

from dataclasses import dataclass
from enum import Enum

from src.data.parsers import ParsedDelivery, ParsedInnings, ParsedMatch
from src.logger import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class ValidationIssue:
    """Represents a single data quality problem found in a match."""
    severity: ValidationSeverity
    field: str           # Which field/section has the issue
    message: str         # Human-readable description

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.field}: {self.message}"

    @property
    def is_error(self) -> bool:
        return self.severity == ValidationSeverity.ERROR

    @property
    def is_warning(self) -> bool:
        return self.severity == ValidationSeverity.WARNING


# ── Match-level validators ────────────────────────────────────────────────────

def _validate_match_meta(match: ParsedMatch) -> list[ValidationIssue]:
    """Validate top-level match metadata."""
    issues = []

    # Match must have a known competition
    if match.competition in ("Unknown", "", None):
        issues.append(ValidationIssue(
            ValidationSeverity.WARNING,
            "competition",
            f"Match {match.cricsheet_file_id} has unknown competition"
        ))

    # Match must have a meaningful date
    if not match.match_date:
        issues.append(ValidationIssue(
            ValidationSeverity.ERROR,
            "match_date",
            "Match has no date"
        ))

    # Teams must be different
    if match.team1 == match.team2:
        issues.append(ValidationIssue(
            ValidationSeverity.ERROR,
            "teams",
            f"Team1 and Team2 are the same: '{match.team1}'"
        ))

    # Toss winner must be one of the two teams
    if match.toss_winner not in (match.team1, match.team2):
        issues.append(ValidationIssue(
            ValidationSeverity.WARNING,
            "toss_winner",
            f"Toss winner '{match.toss_winner}' is not one of the playing teams"
        ))

    # Winner (if set) must be one of the two teams
    if match.winner and match.winner not in (match.team1, match.team2):
        issues.append(ValidationIssue(
            ValidationSeverity.WARNING,
            "winner",
            f"Winner '{match.winner}' is not one of the playing teams"
        ))

    # Win margin consistency
    if match.result == "win":
        if match.winner is None:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                "outcome",
                "Result is 'win' but no winner specified"
            ))
        if match.win_by_runs is None and match.win_by_wickets is None:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "outcome",
                "Result is 'win' but no win margin (runs/wickets) specified"
            ))

    return issues


def _validate_innings_count(match: ParsedMatch) -> list[ValidationIssue]:
    """Validate number of innings."""
    issues = []
    n = len(match.innings)

    if n == 0:
        issues.append(ValidationIssue(
            ValidationSeverity.ERROR,
            "innings",
            "Match has no innings data"
        ))
        return issues

    # Expect 2 innings for limited-overs matches
    if match.match_type in ("T20", "IT20", "ODI") and n < 2:
        # Could be a rain-affected match with only 1 innings
        issues.append(ValidationIssue(
            ValidationSeverity.WARNING,
            "innings",
            f"Limited-overs match has only {n} innings (may be rain-affected)"
        ))

    return issues


# ── Innings-level validators ──────────────────────────────────────────────────

def _validate_innings(innings: ParsedInnings, match_id: str) -> list[ValidationIssue]:
    """Validate a single innings."""
    issues = []
    prefix = f"match {match_id} innings {innings.innings_number}"

    if not innings.deliveries:
        issues.append(ValidationIssue(
            ValidationSeverity.WARNING,
            "deliveries",
            f"{prefix}: No deliveries recorded (match may have been abandoned)"
        ))
        return issues

    # Wickets should not exceed 10
    if innings.total_wickets > 10:
        issues.append(ValidationIssue(
            ValidationSeverity.ERROR,
            "wickets",
            f"{prefix}: {innings.total_wickets} wickets recorded (maximum is 10)"
        ))

    # Check cumulative runs are monotonically increasing
    issues.extend(_validate_cumulative_runs(innings.deliveries, prefix))

    # Check delivery run totals
    issues.extend(_validate_delivery_runs(innings.deliveries, prefix))

    # Check phase assignments cover all overs
    issues.extend(_validate_phases(innings.deliveries, prefix))

    return issues


def _validate_cumulative_runs(
    deliveries: list[ParsedDelivery],
    prefix: str,
) -> list[ValidationIssue]:
    """Cumulative runs must be non-decreasing (can never go backwards)."""
    issues = []
    prev_runs = 0

    for d in deliveries:
        if d.cumulative_runs < prev_runs:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                "cumulative_runs",
                f"{prefix} over {d.over_number}.{d.ball_in_over}: "
                f"cumulative runs decreased from {prev_runs} to {d.cumulative_runs}"
            ))
        prev_runs = d.cumulative_runs

    return issues


def _validate_delivery_runs(
    deliveries: list[ParsedDelivery],
    prefix: str,
) -> list[ValidationIssue]:
    """Individual delivery run values should be in realistic ranges."""
    issues = []

    for d in deliveries:
        # Batter can score 0–6 on a single ball (6 is a six, no more)
        # Exception: overthrows can give more runs, but Cricsheet caps at batter's runs
        if d.runs_batter < 0:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                "runs_batter",
                f"{prefix} over {d.over_number}.{d.ball_in_over}: "
                f"negative batter runs: {d.runs_batter}"
            ))
        if d.runs_batter > 7:
            # Extremely rare but possible (overthrows on a 6)
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "runs_batter",
                f"{prefix} over {d.over_number}.{d.ball_in_over}: "
                f"unusually high batter runs: {d.runs_batter} (possible overthrow)"
            ))

        # Total runs should equal batter + extras
        expected_total = d.runs_batter + d.runs_extras
        if d.runs_total != expected_total:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "runs_total",
                f"{prefix} over {d.over_number}.{d.ball_in_over}: "
                f"runs_total ({d.runs_total}) != batter ({d.runs_batter}) + extras ({d.runs_extras})"
            ))

    return issues


def _validate_phases(
    deliveries: list[ParsedDelivery],
    prefix: str,
) -> list[ValidationIssue]:
    """All deliveries should have a valid phase assigned."""
    issues = []
    valid_phases = {"powerplay", "middle", "death", "full"}

    for d in deliveries:
        if d.phase not in valid_phases:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "phase",
                f"{prefix} over {d.over_number}: unknown phase '{d.phase}'"
            ))

    return issues


# ── Player registry validator ─────────────────────────────────────────────────

def _validate_player_registry(match: ParsedMatch) -> list[ValidationIssue]:
    """
    Check that all players mentioned in deliveries have registry IDs.
    Players not in the registry get their name used as ID (fallback in parser).
    This check helps identify data quality issues in older files.
    """
    issues = []
    registry = match.player_registry

    players_in_deliveries: set[str] = set()
    for inn in match.innings:
        for d in inn.deliveries:
            players_in_deliveries.update([
                d.batter_name, d.bowler_name, d.non_striker_name
            ])
            if d.player_out_name:
                players_in_deliveries.add(d.player_out_name)

    missing = players_in_deliveries - set(registry.keys())
    if missing:
        issues.append(ValidationIssue(
            ValidationSeverity.WARNING,
            "player_registry",
            f"Match {match.cricsheet_file_id}: {len(missing)} player(s) not in registry: "
            f"{', '.join(sorted(missing)[:5])}{'...' if len(missing) > 5 else ''}"
        ))

    return issues


# ── Main validation entry point ───────────────────────────────────────────────

def validate_parsed_match(match: ParsedMatch) -> list[ValidationIssue]:
    """
    Run all validators on a parsed match.

    Args:
        match: ParsedMatch from the parser.

    Returns:
        List of ValidationIssue objects. Empty list = no issues found.
        Call .is_error on each to check severity.

    Usage:
        issues = validate_parsed_match(parsed_match)
        errors = [i for i in issues if i.is_error]
        if errors:
            logger.error("Skipping match %s: %s", match.cricsheet_file_id, errors[0])
        else:
            # proceed to database insertion
    """
    all_issues: list[ValidationIssue] = []

    # Match-level checks
    all_issues.extend(_validate_match_meta(match))
    all_issues.extend(_validate_innings_count(match))
    all_issues.extend(_validate_player_registry(match))

    # Innings-level checks (only if innings exist)
    for innings in match.innings:
        all_issues.extend(_validate_innings(innings, match.cricsheet_file_id))

    # Log summary
    errors = [i for i in all_issues if i.is_error]
    warnings = [i for i in all_issues if i.is_warning]

    if errors:
        logger.error(
            "Match %s: %d error(s), %d warning(s)",
            match.cricsheet_file_id, len(errors), len(warnings)
        )
    elif warnings:
        logger.debug(
            "Match %s: 0 errors, %d warning(s)",
            match.cricsheet_file_id, len(warnings)
        )

    return all_issues


def has_blocking_errors(issues: list[ValidationIssue]) -> bool:
    """
    Returns True if any issues are errors (should block ingestion).
    Warnings alone do not block ingestion.
    """
    return any(i.is_error for i in issues)
