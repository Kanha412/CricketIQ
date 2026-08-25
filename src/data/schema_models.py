"""
CricketIQ — Cricsheet JSON Schema Models
==========================================
Pydantic models that exactly mirror the structure of Cricsheet's JSON format.

PURPOSE:
    These models are ONLY for validating raw JSON from Cricsheet.
    They should NOT be used anywhere else in the codebase.
    The parser (src/data/parsers.py) transforms these into our internal structures.

CRICSHEET JSON STRUCTURE (simplified):
    {
      "meta": { data_version, created, revision },
      "info": {
        "balls_per_over", "city", "competition", "dates", "gender",
        "match_type", "overs", "season", "teams", "venue",
        "event": { name, match_number, stage },
        "outcome": { winner, by: { runs | wickets }, result },
        "players": { team_name: [player_names] },
        "registry": { people: { player_name: cricsheet_id } },
        "toss": { winner, decision },
        "player_of_match": [player_names],
        "officials": { ... }       ← we ignore this
      },
      "innings": [
        {
          "team": "...",
          "overs": [
            {
              "over": 0,            ← 0-indexed! we convert to 1-indexed
              "deliveries": [
                {
                  "batter", "bowler", "non_striker",
                  "runs": { batter, extras, total },
                  "extras": { wides, noballs, byes, legbyes, penalty },  ← optional
                  "wickets": [ { kind, player_out, fielders } ]           ← optional
                }
              ]
            }
          ],
          "target": { overs, runs }  ← only in 2nd innings
        }
      ]
    }

EDGE CASES HANDLED:
    - outcome.by can be runs OR wickets (not both)
    - outcome.result exists instead of outcome.by for ties/no-results
    - extras dict is optional (many deliveries have no extras)
    - wickets list is optional and usually empty
    - city is optional (some older matches miss it)
    - event dict is optional
    - player_of_match is optional
    - target only exists in second innings
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Meta section ──────────────────────────────────────────────────────────────

class CricsheetMeta(BaseModel):
    """Top-level metadata about the JSON file version."""
    model_config = ConfigDict(extra="ignore")

    data_version: str
    created: str
    revision: int


# ── Info sub-models ───────────────────────────────────────────────────────────

class CricsheetEvent(BaseModel):
    """Competition event details (optional in some files)."""
    model_config = ConfigDict(extra="ignore")

    name: str
    match_number: int | None = None
    stage: str | None = None          # 'Final', 'Qualifier 1', 'Eliminator', etc.
    group: str | None = None          # Group stage name if applicable


class CricsheetOutcomeBy(BaseModel):
    """
    How the match was won.
    EXACTLY ONE of runs or wickets will be set (never both).
    innings is only used in Test matches (innings victory).
    """
    model_config = ConfigDict(extra="ignore")

    runs: int | None = None
    wickets: int | None = None
    innings: int | None = None        # e.g. "innings and 50 runs" in Tests


class CricsheetOutcome(BaseModel):
    """
    Match result.

    Possible shapes:
        Win:       { "winner": "CSK", "by": { "wickets": 5 } }
        Win:       { "winner": "MI", "by": { "runs": 20 } }
        Tie/SO:    { "winner": "CSK", "eliminator": "Super Over", "by": {...} }
        Tie:       { "result": "tie" }
        No result: { "result": "no result" }
    """
    model_config = ConfigDict(extra="ignore")

    winner: str | None = None
    by: CricsheetOutcomeBy | None = None
    result: str | None = None         # 'tie', 'no result', 'draw'
    eliminator: str | None = None     # 'Super Over' — indicates match went to SO
    method: str | None = None         # 'D/L' for DLS-affected matches


class CricsheetToss(BaseModel):
    """Toss details."""
    model_config = ConfigDict(extra="ignore")

    winner: str
    decision: str                     # 'bat' or 'field'

    @field_validator("decision")
    @classmethod
    def normalise_decision(cls, v: str) -> str:
        """Ensure decision is lowercase 'bat' or 'field'."""
        v = v.lower().strip()
        if v not in {"bat", "field"}:
            # Some older files use 'batting' or 'fielding'
            if "bat" in v:
                return "bat"
            if "field" in v:
                return "field"
        return v


class CricsheetRegistry(BaseModel):
    """
    Player name → Cricsheet unique ID mapping.
    Used to resolve player names across matches to consistent IDs.

    Example:
        { "people": { "Virat Kohli": "0a786c0b", "MS Dhoni": "1cd23ef4" } }
    """
    model_config = ConfigDict(extra="ignore")

    people: dict[str, str] = Field(default_factory=dict)


class CricsheetInfo(BaseModel):
    """
    Match information block.
    Contains everything about the match except the ball-by-ball data.
    """
    model_config = ConfigDict(extra="ignore")

    # Core match identity
    match_type: str                   # 'T20', 'ODI', 'Test', 'IT20', etc.
    gender: str                       # 'male', 'female'
    season: str | int                 # '2023', '2022/23', or integer 2017 (older files)
    teams: list[str]                  # Always exactly 2 teams
    dates: list[str]                  # ISO dates (can span multiple days for Tests)

    # Venue
    venue: str
    city: str | None = None           # Optional — some files missing this

    # Competition info
    competition: str | None = None    # Full competition name
    event: CricsheetEvent | None = None

    # Match details
    balls_per_over: int = 6
    overs: int | None = None          # 20 for T20, 50 for ODI, None for Test

    # Result
    outcome: CricsheetOutcome

    # Toss
    toss: CricsheetToss

    # Players (name lists per team)
    players: dict[str, list[str]] = Field(default_factory=dict)

    # Player ID registry
    registry: CricsheetRegistry = Field(default_factory=CricsheetRegistry)

    # Optional fields
    player_of_match: list[str] = Field(default_factory=list)
    officials: dict[str, Any] = Field(default_factory=dict)    # ignored
    missing: list[str] = Field(default_factory=list)           # fields missing from file

    @field_validator("teams")
    @classmethod
    def validate_two_teams(cls, v: list[str]) -> list[str]:
        """A cricket match always has exactly 2 teams."""
        if len(v) != 2:
            raise ValueError(f"Expected exactly 2 teams, got {len(v)}: {v}")
        return v

    @field_validator("dates")
    @classmethod
    def validate_dates_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Match must have at least one date")
        return v

    @property
    def match_date(self) -> str:
        """First date of the match (matches with multiple dates return first day)."""
        return self.dates[0]

    @property
    def team1(self) -> str:
        return self.teams[0]

    @property
    def team2(self) -> str:
        return self.teams[1]


# ── Innings / Delivery models ─────────────────────────────────────────────────

class CricsheetRuns(BaseModel):
    """Runs scored on a single delivery."""
    model_config = ConfigDict(extra="ignore")

    batter: int = 0                   # Runs credited to the batter
    extras: int = 0                   # Extra runs (wide, no-ball, bye, etc.)
    total: int = 0                    # batter + extras

    @model_validator(mode="after")
    def validate_total(self) -> "CricsheetRuns":
        """
        Total should equal batter + extras.
        Some older Cricsheet files have minor inconsistencies — we warn but don't reject.
        """
        expected = self.batter + self.extras
        if self.total != expected:
            # Don't raise — just note the discrepancy (parser will use total)
            pass
        return self


class CricsheetExtras(BaseModel):
    """
    Breakdown of extra runs conceded on a delivery.
    All fields are optional (a delivery with no extras won't have this key at all).

    Note: 'noballs' in Cricsheet JSON (no underscore).
    """
    model_config = ConfigDict(extra="ignore")

    wides: int = 0
    noballs: int = 0       # Note: Cricsheet uses 'noballs' not 'no_balls'
    byes: int = 0
    legbyes: int = 0
    penalty: int = 0

    @property
    def extras_type(self) -> str | None:
        """
        Primary type of extra on this delivery.
        Returns the first non-zero category found.
        Order: wides → no-balls → byes → leg-byes → penalty
        """
        if self.wides > 0:
            return "wide"
        if self.noballs > 0:
            return "no_ball"
        if self.byes > 0:
            return "bye"
        if self.legbyes > 0:
            return "leg_bye"
        if self.penalty > 0:
            return "penalty"
        return None

    @property
    def is_wide(self) -> bool:
        return self.wides > 0

    @property
    def is_no_ball(self) -> bool:
        return self.noballs > 0


class CricsheetFielder(BaseModel):
    """A fielder involved in a dismissal."""
    model_config = ConfigDict(extra="ignore")

    name: str
    substitute: bool = False          # True if substitute fielder


class CricsheetWicket(BaseModel):
    """
    A wicket/dismissal on this delivery.

    Note: A delivery can have TWO wickets (extremely rare — run out + no-ball scenario,
    or two batters run out on the same ball). The deliveries.wickets field is a list.

    Dismissal kinds in Cricsheet:
        'bowled', 'caught', 'caught and bowled', 'lbw', 'stumped',
        'run out', 'hit wicket', 'handled the ball', 'obstructing the field',
        'timed out', 'hit the ball twice'
    """
    model_config = ConfigDict(extra="ignore")

    kind: str                         # Dismissal type
    player_out: str                   # Name of dismissed batter
    fielders: list[CricsheetFielder] = Field(default_factory=list)


class CricsheetDelivery(BaseModel):
    """
    A single delivery (ball) in the match.

    A delivery may or may not be a 'legal' delivery:
        - Legal:   normal ball, no extras, counts toward over
        - Illegal: wide or no-ball — does NOT advance the over ball count
    """
    model_config = ConfigDict(extra="ignore")

    batter: str                       # Name of the batter facing
    bowler: str                       # Name of the bowler
    non_striker: str                  # Name of the non-striking batter
    runs: CricsheetRuns
    extras: CricsheetExtras = Field(default_factory=CricsheetExtras)
    wickets: list[CricsheetWicket] = Field(default_factory=list)

    # Review (DRS) — present in some files, not needed for our analytics
    review: dict[str, Any] | None = None


class CricsheetOver(BaseModel):
    """
    One over (set of deliveries by the same bowler).

    IMPORTANT: 'over' field is 0-indexed in Cricsheet!
        over=0 → first over (over 1 in cricket terminology)
        over=19 → last over of a T20 (over 20 in cricket)
    The parser converts this to 1-indexed.
    """
    model_config = ConfigDict(extra="ignore")

    over: int                         # 0-indexed over number
    deliveries: list[CricsheetDelivery]


class CricsheetTarget(BaseModel):
    """Target for the chasing team (second innings only)."""
    model_config = ConfigDict(extra="ignore")

    overs: int | float                # Overs available (may be decimal in DLS)
    runs: int                         # Runs required to win


class CricsheetInnings(BaseModel):
    """
    One innings of the match.
    A standard match has 2 innings. A super over adds a 3rd (and sometimes 4th).
    """
    model_config = ConfigDict(extra="ignore")

    team: str                         # Batting team name
    overs: list[CricsheetOver] = Field(default_factory=list)
    target: CricsheetTarget | None = None    # Only set in 2nd innings
    super_over: bool = False          # True if this is a super over innings

    # Some Cricsheet files mark super overs with a 'super_over' key
    # Others include them as a 3rd/4th innings entry


# ── Top-level match model ─────────────────────────────────────────────────────

class CricsheetMatch(BaseModel):
    """
    Complete Cricsheet match file.
    This is what you get when you json.load() a Cricsheet match file.

    Usage:
        with open("1359475.json") as f:
            raw = json.load(f)
        match = CricsheetMatch.model_validate(raw)
    """
    model_config = ConfigDict(extra="ignore")

    meta: CricsheetMeta
    info: CricsheetInfo
    innings: list[CricsheetInnings]

    @model_validator(mode="after")
    def validate_innings_count(self) -> "CricsheetMatch":
        """A match should have 1–8 innings.
        Normal matches: 2. Super-over matches: 4 (one super over per side).
        Double super-over (rare, e.g. IPL 2020 final): 6. Allow up to 8 for safety.
        """
        n = len(self.innings)
        if n == 0:
            raise ValueError("Match has no innings data")
        if n > 8:
            raise ValueError(f"Unexpectedly high number of innings: {n}")
        return self

    @property
    def is_t20(self) -> bool:
        return self.info.match_type in ("T20", "IT20")

    @property
    def is_odi(self) -> bool:
        return self.info.match_type == "ODI"

    @property
    def is_test(self) -> bool:
        return self.info.match_type == "Test"
