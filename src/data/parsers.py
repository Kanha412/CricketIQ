"""
CricketIQ — Cricsheet JSON Parser
===================================
Transforms validated CricsheetMatch objects into our internal data structures.

WHAT THIS FILE DOES:
    Takes a CricsheetMatch (validated Pydantic model) and produces:
        - ParsedMatch       : flat match-level info ready for DB insertion
        - ParsedInnings     : innings-level info
        - ParsedDelivery    : one row per delivery, with all derived fields

WHY SEPARATE FROM SCHEMA MODELS:
    Schema models (schema_models.py) mirror Cricsheet's world.
    This file creates OUR world — different naming, derived fields, 1-indexed overs.

KEY TRANSFORMATIONS:
    1. over_number: 0-indexed (Cricsheet) → 1-indexed (ours)
    2. ball_in_over: counted per over including wides/no-balls
    3. legal_ball_number: only counts legal deliveries (advances 1 per legal ball)
    4. phase: computed from over_number based on match_type
    5. is_legal_delivery: False for wides and no-balls
    6. is_boundary: True if batter scored 4 or 6
    7. is_six: True if batter scored 6
    8. is_dot_ball: True if legal delivery and zero runs total
    9. cumulative_runs: running total within innings
    10. cumulative_wickets: running wicket count within innings
    11. player names → cricsheet_ids (from registry)

DESIGN DECISION:
    ParsedDelivery uses a @dataclass (not Pydantic) because:
    - It's an internal structure, not a validation boundary
    - Dataclasses are lighter and faster for thousands of rows
    - Pydantic is used at the boundary (loading raw JSON)
"""

from dataclasses import dataclass, field
from datetime import date

from src.data.schema_models import (
    CricsheetDelivery,
    CricsheetInnings,
    CricsheetMatch,
)
from src.logger import get_logger

logger = get_logger(__name__)


# ── Phase Boundaries (by match type) ─────────────────────────────────────────

PHASE_BOUNDARIES: dict[str, dict[str, tuple[int, int]]] = {
    "T20": {
        "powerplay": (1, 6),
        "middle":    (7, 15),
        "death":     (16, 20),
    },
    "IT20": {  # T20 Internationals — same boundaries
        "powerplay": (1, 6),
        "middle":    (7, 15),
        "death":     (16, 20),
    },
    "ODI": {
        "powerplay": (1, 10),
        "middle":    (11, 40),
        "death":     (41, 50),
    },
    "Test": {
        "full": (1, 999),           # Tests have no phases — assign 'full'
    },
}


def get_phase(over_number: int, match_type: str) -> str:
    """
    Assign a phase label to a delivery based on the over number.

    Args:
        over_number: 1-indexed over number.
        match_type:  'T20', 'IT20', 'ODI', 'Test'.

    Returns:
        Phase label: 'powerplay', 'middle', 'death', or 'full'.
    """
    boundaries = PHASE_BOUNDARIES.get(match_type, PHASE_BOUNDARIES["T20"])
    for phase_name, (start, end) in boundaries.items():
        if start <= over_number <= end:
            return phase_name
    # Fallback — shouldn't happen in clean data
    return "death" if over_number > 15 else "middle"


# ── Internal Data Structures (what we store in the DB) ───────────────────────

@dataclass
class ParsedDelivery:
    """
    One row in the deliveries table.

    Fields prefixed with _ are derived (computed during parsing, not in raw JSON).
    All player references use Cricsheet IDs (from registry), not names.
    """
    # ── Positional ─────────────────────────────────────────────────────────
    innings_number: int           # 1 or 2 (3/4 for super overs)
    over_number: int              # 1-indexed (Cricsheet is 0-indexed)
    ball_in_over: int             # Delivery sequence within over (1, 2, 3...)
                                  # Includes wides/no-balls in the count
    legal_ball_number: int        # Legal ball count (1-6), ignores wides/no-balls

    # ── Players (Cricsheet IDs) ─────────────────────────────────────────────
    batter_id: str                # Cricsheet registry ID
    batter_name: str              # Original name (for DB lookup / creation)
    non_striker_id: str
    non_striker_name: str
    bowler_id: str
    bowler_name: str

    # ── Runs ───────────────────────────────────────────────────────────────
    runs_batter: int              # Runs credited to the batter
    runs_extras: int              # Extra runs (wide, no-ball, bye, etc.)
    runs_total: int               # batter + extras

    # ── Extras ─────────────────────────────────────────────────────────────
    extras_type: str | None       # 'wide', 'no_ball', 'bye', 'leg_bye', 'penalty', None

    # ── Wickets ────────────────────────────────────────────────────────────
    is_wicket: bool               # True if at least one wicket fell on this ball
    wicket_kind: str | None       # 'caught', 'bowled', 'run out', etc.
    player_out_id: str | None     # Cricsheet ID of dismissed player
    player_out_name: str | None   # Name of dismissed player
    fielder_names: list[str] = field(default_factory=list)   # Names of fielders involved

    # ── Derived fields (computed at parse time) ─────────────────────────────
    phase: str = ""               # 'powerplay', 'middle', 'death'
    is_legal_delivery: bool = True    # False for wides and no-balls
    is_boundary: bool = False         # True if batter scored 4 or 6
    is_six: bool = False              # True if batter scored 6
    is_dot_ball: bool = False         # True if legal delivery with 0 total runs
    cumulative_runs: int = 0          # Running total within innings at this point
    cumulative_wickets: int = 0       # Running wicket count at this point


@dataclass
class ParsedInnings:
    """One innings of the match."""
    innings_number: int           # 1-indexed
    batting_team: str             # Team name (batting)
    bowling_team: str             # Team name (bowling)
    target_runs: int | None       # Only set in 2nd innings
    is_super_over: bool
    deliveries: list[ParsedDelivery] = field(default_factory=list)

    @property
    def total_runs(self) -> int:
        """Sum of all runs in the innings."""
        return sum(d.runs_total for d in self.deliveries)

    @property
    def total_wickets(self) -> int:
        """Total wickets fallen."""
        return sum(1 for d in self.deliveries if d.is_wicket)

    @property
    def legal_deliveries(self) -> int:
        """Count of legal deliveries bowled."""
        return sum(1 for d in self.deliveries if d.is_legal_delivery)


@dataclass
class ParsedMatch:
    """
    Complete parsed match — everything needed to insert into the database.
    Flat structure at match level; innings and deliveries are nested.
    """
    # ── Identity ────────────────────────────────────────────────────────────
    cricsheet_file_id: str        # Filename without .json (e.g. '1359475')
    competition: str              # 'Indian Premier League'
    season: str                   # '2023'
    match_type: str               # 'T20', 'ODI', 'Test'
    gender: str                   # 'male', 'female'

    # ── Date / Venue ────────────────────────────────────────────────────────
    match_date: str               # ISO date string of first day
    venue: str
    city: str | None

    # ── Teams ───────────────────────────────────────────────────────────────
    team1: str                    # teams[0] from Cricsheet
    team2: str                    # teams[1] from Cricsheet

    # ── Toss ────────────────────────────────────────────────────────────────
    toss_winner: str
    toss_decision: str            # 'bat' or 'field'

    # ── Result ──────────────────────────────────────────────────────────────
    winner: str | None            # None if no result or tie
    win_by_runs: int | None
    win_by_wickets: int | None
    result: str                   # 'win', 'tie', 'no_result', 'draw'
    dls_applied: bool             # True if D/L method was used

    # ── Event ───────────────────────────────────────────────────────────────
    event_name: str | None        # Competition event name
    match_number: int | None      # Match number within tournament
    stage: str | None             # 'Final', 'Qualifier 1', etc.

    # ── Players ─────────────────────────────────────────────────────────────
    player_of_match: list[str]    # Names of player(s) of the match
    player_registry: dict[str, str]   # name → cricsheet_id (from info.registry)
    team_players: dict[str, list[str]]  # team_name → [player_names]

    # ── Innings ─────────────────────────────────────────────────────────────
    innings: list[ParsedInnings] = field(default_factory=list)


# ── Core Parser ───────────────────────────────────────────────────────────────

class CricsheetParser:
    """
    Parses a CricsheetMatch into our internal ParsedMatch structure.

    Usage:
        parser = CricsheetParser()
        parsed = parser.parse(cricsheet_match, file_id="1359475")
    """

    def parse(self, match: CricsheetMatch, file_id: str) -> ParsedMatch:
        """
        Main entry point. Parse a validated CricsheetMatch.

        Args:
            match:    Validated CricsheetMatch Pydantic model.
            file_id:  Match file ID (filename without .json).

        Returns:
            ParsedMatch with all derived fields computed.
        """
        logger.debug("Parsing match %s (%s %s)", file_id, match.info.match_type, match.info.dates[0])

        info = match.info
        registry = info.registry.people      # name → cricsheet_id

        # Parse match result
        winner, win_by_runs, win_by_wickets, result, dls = self._parse_outcome(info.outcome)

        # Parse innings
        # Identify the bowling team for each innings
        parsed_innings = self._parse_all_innings(
            raw_innings=match.innings,
            teams=[info.team1, info.team2],
            match_type=info.match_type,
            registry=registry,
        )

        return ParsedMatch(
            cricsheet_file_id=file_id,
            competition=info.competition or "Unknown",
            season=str(info.season),          # normalise int → str for older files
            match_type=info.match_type,
            gender=info.gender,
            match_date=info.match_date,
            venue=info.venue,
            city=info.city,
            team1=info.team1,
            team2=info.team2,
            toss_winner=info.toss.winner,
            toss_decision=info.toss.decision,
            winner=winner,
            win_by_runs=win_by_runs,
            win_by_wickets=win_by_wickets,
            result=result,
            dls_applied=dls,
            event_name=info.event.name if info.event else None,
            match_number=info.event.match_number if info.event else None,
            stage=info.event.stage if info.event else None,
            player_of_match=info.player_of_match,
            player_registry=registry,
            team_players=info.players,
            innings=parsed_innings,
        )

    # ── Outcome parsing ───────────────────────────────────────────────────────

    def _parse_outcome(
        self,
        outcome: "CricsheetOutcome",  # type: ignore[name-defined]  # noqa: F821
    ) -> tuple[str | None, int | None, int | None, str, bool]:
        """
        Parse the match outcome into flat fields.

        Returns:
            (winner, win_by_runs, win_by_wickets, result_str, dls_applied)
        """
        winner = outcome.winner
        win_by_runs = None
        win_by_wickets = None
        dls = outcome.method is not None and "d/l" in outcome.method.lower()

        if outcome.result:
            # No winner — tie or no result
            result_map = {
                "tie": "tie",
                "no result": "no_result",
                "draw": "draw",
            }
            result = result_map.get(outcome.result.lower(), "no_result")
            return None, None, None, result, dls

        if outcome.by:
            win_by_runs = outcome.by.runs
            win_by_wickets = outcome.by.wickets

        result = "win" if winner else "no_result"
        return winner, win_by_runs, win_by_wickets, result, dls

    # ── Innings parsing ───────────────────────────────────────────────────────

    def _parse_all_innings(
        self,
        raw_innings: list[CricsheetInnings],
        teams: list[str],
        match_type: str,
        registry: dict[str, str],
    ) -> list[ParsedInnings]:
        """
        Parse all innings in the match.

        Determines bowling team by process of elimination:
            - Team batting in innings 1 → other team is bowling
            - For super overs, same logic applies
        """
        parsed = []

        for i, raw_inn in enumerate(raw_innings):
            innings_number = i + 1
            batting_team = raw_inn.team

            # Bowling team = whichever team is NOT batting
            bowling_team = next((t for t in teams if t != batting_team), teams[0])

            # A 3rd/4th innings is a super over
            is_super_over = innings_number > 2

            parsed_inn = self._parse_innings(
                raw_inn=raw_inn,
                innings_number=innings_number,
                batting_team=batting_team,
                bowling_team=bowling_team,
                match_type=match_type,
                registry=registry,
                is_super_over=is_super_over,
            )
            parsed.append(parsed_inn)

        return parsed

    def _parse_innings(
        self,
        raw_inn: CricsheetInnings,
        innings_number: int,
        batting_team: str,
        bowling_team: str,
        match_type: str,
        registry: dict[str, str],
        is_super_over: bool,
    ) -> ParsedInnings:
        """Parse a single innings, computing derived delivery fields."""

        target_runs = raw_inn.target.runs if raw_inn.target else None

        parsed_inn = ParsedInnings(
            innings_number=innings_number,
            batting_team=batting_team,
            bowling_team=bowling_team,
            target_runs=target_runs,
            is_super_over=is_super_over,
        )

        # Running counters (reset per innings)
        cumulative_runs = 0
        cumulative_wickets = 0

        for raw_over in raw_inn.overs:
            # Convert 0-indexed → 1-indexed
            over_number = raw_over.over + 1
            phase = get_phase(over_number, match_type)

            # Track ball counts within this over
            ball_sequence = 0          # Every delivery including extras
            legal_ball_count = 0       # Only legal deliveries

            for raw_delivery in raw_over.deliveries:
                ball_sequence += 1

                # Parse derived boolean fields
                is_legal = self._is_legal_delivery(raw_delivery)
                if is_legal:
                    legal_ball_count += 1

                is_boundary = raw_delivery.runs.batter in (4, 6)
                is_six = raw_delivery.runs.batter == 6
                is_dot = is_legal and raw_delivery.runs.total == 0

                # Update running totals
                cumulative_runs += raw_delivery.runs.total
                if raw_delivery.wickets:
                    cumulative_wickets += len(raw_delivery.wickets)

                # Parse wicket info (take the first wicket if multiple)
                is_wicket = bool(raw_delivery.wickets)
                wicket_kind = None
                player_out_name = None
                player_out_id = None
                fielder_names = []

                if raw_delivery.wickets:
                    first_wicket = raw_delivery.wickets[0]
                    wicket_kind = first_wicket.kind
                    player_out_name = first_wicket.player_out
                    player_out_id = registry.get(player_out_name, player_out_name)
                    fielder_names = [f.name for f in first_wicket.fielders]

                # Resolve player names → Cricsheet IDs
                batter_name = raw_delivery.batter
                bowler_name = raw_delivery.bowler
                non_striker_name = raw_delivery.non_striker

                batter_id = registry.get(batter_name, batter_name)
                bowler_id = registry.get(bowler_name, bowler_name)
                non_striker_id = registry.get(non_striker_name, non_striker_name)

                # Log warning if player not in registry (data quality issue)
                if batter_id == batter_name and batter_name not in registry:
                    logger.debug(
                        "Player not in registry: '%s' (innings %d, over %d)",
                        batter_name, innings_number, over_number
                    )

                parsed_delivery = ParsedDelivery(
                    innings_number=innings_number,
                    over_number=over_number,
                    ball_in_over=ball_sequence,
                    legal_ball_number=legal_ball_count,
                    batter_id=batter_id,
                    batter_name=batter_name,
                    non_striker_id=non_striker_id,
                    non_striker_name=non_striker_name,
                    bowler_id=bowler_id,
                    bowler_name=bowler_name,
                    runs_batter=raw_delivery.runs.batter,
                    runs_extras=raw_delivery.runs.extras,
                    runs_total=raw_delivery.runs.total,
                    extras_type=raw_delivery.extras.extras_type,
                    is_wicket=is_wicket,
                    wicket_kind=wicket_kind,
                    player_out_id=player_out_id,
                    player_out_name=player_out_name,
                    fielder_names=fielder_names,
                    phase=phase,
                    is_legal_delivery=is_legal,
                    is_boundary=is_boundary,
                    is_six=is_six,
                    is_dot_ball=is_dot,
                    cumulative_runs=cumulative_runs,
                    cumulative_wickets=cumulative_wickets,
                )
                parsed_inn.deliveries.append(parsed_delivery)

        logger.debug(
            "Innings %d parsed: %d deliveries, %d runs, %d wickets",
            innings_number,
            len(parsed_inn.deliveries),
            parsed_inn.total_runs,
            parsed_inn.total_wickets,
        )
        return parsed_inn

    # ── Helper methods ────────────────────────────────────────────────────────

    @staticmethod
    def _is_legal_delivery(delivery: CricsheetDelivery) -> bool:
        """
        A delivery is legal if it is NOT a wide and NOT a no-ball.
        Legal deliveries advance the ball count within the over (1 → 6).
        Byes and leg-byes ARE legal deliveries (ball counts, batter just doesn't score).
        """
        return not delivery.extras.is_wide and not delivery.extras.is_no_ball


# ── Convenience function ──────────────────────────────────────────────────────

def parse_match_file(file_path: str | "Path", file_id: str | None = None) -> ParsedMatch:  # type: ignore[name-defined]  # noqa: F821
    """
    Load and parse a Cricsheet JSON file in one step.

    Args:
        file_path:  Path to the .json file.
        file_id:    Match ID override. If None, inferred from filename.

    Returns:
        ParsedMatch ready for database insertion.

    Raises:
        FileNotFoundError:  If the file doesn't exist.
        ValueError:         If the JSON is invalid or fails Pydantic validation.
    """
    import json
    from pathlib import Path

    from src.data.schema_models import CricsheetMatch
    from src.data.validators import validate_parsed_match

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Match file not found: {path}")

    if file_id is None:
        file_id = path.stem    # filename without .json

    with open(path, encoding="utf-8") as f:
        raw_data = json.load(f)

    # Step 1: Validate JSON structure (Pydantic)
    match = CricsheetMatch.model_validate(raw_data)

    # Step 2: Parse into internal structure
    parser = CricsheetParser()
    parsed = parser.parse(match, file_id=file_id)

    # Step 3: Business logic validation
    issues = validate_parsed_match(parsed)
    if issues:
        for issue in issues:
            logger.warning("Validation issue in %s: %s", file_id, issue)

    return parsed
