"""
CricketIQ — AI Agent (Gemini with function calling)
=====================================================
This module implements the conversation loop:
  1. User message → Gemini (with tool declarations)
  2. Gemini emits a function_call → we dispatch to TOOL_FUNCTIONS
  3. Tool result → back to Gemini
  4. Gemini emits text → return to UI
  5. Repeat (multi-turn conversation history preserved)

WHY GEMINI:
    Free-tier access (Gemini 2.0 Flash) supports function calling.
    We use google-generativeai SDK which is already in pyproject.toml.

GRACEFUL FALLBACK:
    If LLM_API_KEY == "test-key-placeholder" or the import fails,
    the agent returns a clear setup message instead of crashing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from src.ai.prompts import SYSTEM_PROMPT
from src.ai.tools import TOOL_FUNCTIONS
from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Gemini SDK import  (graceful if not installed or key is wrong)
# ---------------------------------------------------------------------------

try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Run: uv add google-generativeai")


# ---------------------------------------------------------------------------
# Gemini function declarations (translates Python docstrings → API schema)
# ---------------------------------------------------------------------------

def _make_function_declarations() -> list[dict]:
    """
    Hand-written Gemini function declarations for each tool.
    Gemini's function calling API requires explicit JSON schema per function.
    """
    return [
        {
            "name": "get_batting_stats",
            "description": "Get career or season batting statistics for an IPL player. Use this before quoting any batting number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "Player name (partial OK, e.g. 'Kohli')"},
                    "season":      {"type": "string", "description": "IPL season e.g. '2023/24' — omit for career stats"},
                },
                "required": ["player_name"],
            },
        },
        {
            "name": "get_bowling_stats",
            "description": "Get career or season bowling statistics for an IPL player. Use this before quoting any bowling number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "Player name (partial OK, e.g. 'Bumrah')"},
                    "season":      {"type": "string", "description": "IPL season e.g. '2023/24' — omit for career stats"},
                },
                "required": ["player_name"],
            },
        },
        {
            "name": "get_matchup",
            "description": "Get head-to-head batter vs bowler statistics. Shows SR, economy, dot%, dominance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "batter_name": {"type": "string", "description": "Batter's name"},
                    "bowler_name": {"type": "string", "description": "Bowler's name"},
                },
                "required": ["batter_name", "bowler_name"],
            },
        },
        {
            "name": "get_team_head_to_head",
            "description": "Get win/loss record between two IPL teams.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_a": {"type": "string", "description": "First team, e.g. 'Mumbai Indians'"},
                    "team_b": {"type": "string", "description": "Second team, e.g. 'Chennai Super Kings'"},
                },
                "required": ["team_a", "team_b"],
            },
        },
        {
            "name": "predict_win_probability",
            "description": "Predict batting-team win probability for an ongoing T20 chase. Use for match-state questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "runs_scored":     {"type": "integer", "description": "Runs scored so far in 2nd innings"},
                    "wickets_fallen":  {"type": "integer", "description": "Wickets lost (0–9)"},
                    "balls_remaining": {"type": "integer", "description": "Legal balls left (0–120)"},
                    "required_runs":   {"type": "integer", "description": "Runs still needed to win"},
                    "phase":           {"type": "string", "enum": ["powerplay", "middle", "death"],
                                        "description": "Current match phase"},
                },
                "required": ["runs_scored", "wickets_fallen", "balls_remaining", "required_runs", "phase"],
            },
        },
        {
            "name": "simulate_innings_score",
            "description": "Run Monte Carlo simulation to estimate score distribution for a T20 innings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "balls_bowled":   {"type": "integer", "description": "Balls already bowled (0 = fresh innings)"},
                    "runs_so_far":    {"type": "integer", "description": "Runs already scored"},
                    "wickets_fallen": {"type": "integer", "description": "Wickets already lost"},
                    "target":         {"type": "integer", "description": "Chase target — omit for 1st innings"},
                    "n_simulations":  {"type": "integer", "description": "Number of simulations (default 5000)"},
                },
                "required": [],
            },
        },
        {
            "name": "search_player_name",
            "description": "Search for IPL players by name. Use when unsure of exact spelling before calling other tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Partial player name to search"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_venue_stats",
            "description": "Get batting/bowling-first win rates for an IPL venue. Use for toss-decision questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "venue_name": {"type": "string", "description": "Venue name or partial name, e.g. 'Wankhede'"},
                },
                "required": ["venue_name"],
            },
        },
    ]


# ---------------------------------------------------------------------------
# Message history management
# ---------------------------------------------------------------------------

@dataclass
class ConversationHistory:
    """Stores the multi-turn message history for a session."""
    messages: list[dict] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "parts": [{"text": text}]})

    def add_model(self, text: str) -> None:
        self.messages.append({"role": "model", "parts": [{"text": text}]})

    def add_tool_result(self, fn_name: str, result: dict) -> None:
        """Append a function-response turn (role=user, functionResponse part)."""
        self.messages.append({
            "role": "user",
            "parts": [{
                "functionResponse": {
                    "name": fn_name,
                    "response": {"result": result},
                }
            }],
        })

    def to_gemini_history(self) -> list[dict]:
        """Return history in the format Gemini expects."""
        return self.messages


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class CricketIQAgent:
    """
    Stateful AI agent for IPL cricket analytics.

    Each Streamlit session creates one instance (via st.session_state).
    The agent holds the full conversation history so Gemini has context.

    Usage:
        agent = CricketIQAgent.create()
        reply, tool_log = agent.chat("How is Rohit Sharma's powerplay SR?")
    """

    MAX_TOOL_ROUNDS = 5   # safety: stop if Gemini keeps calling tools indefinitely

    def __init__(self, model: Any, history: ConversationHistory) -> None:
        self._model   = model
        self._history = history

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls) -> "CricketIQAgent":
        """
        Create a new agent.  Raises RuntimeError if the API key is missing
        or is still the placeholder value (so the UI can show setup help).
        """
        api_key = os.getenv("LLM_API_KEY", "")
        if not api_key or api_key == "test-key-placeholder":
            raise RuntimeError("placeholder_key")

        if not _GENAI_AVAILABLE:
            raise RuntimeError("genai_not_installed")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name    = "gemini-2.0-flash",
            system_instruction = SYSTEM_PROMPT,
            tools         = [{"function_declarations": _make_function_declarations()}],
        )
        return cls(model=model, history=ConversationHistory())

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> tuple[str, list[dict]]:
        """
        Send a user message and return (assistant_text, tool_log).

        tool_log is a list of dicts:
            { "tool": fn_name, "args": {...}, "result": {...} }
        Useful for the UI to show "Queried get_batting_stats..." transparency.

        Returns ("", []) is never the case — on error we return the error message.
        """
        tool_log: list[dict] = []
        self._history.add_user(user_message)

        chat_session = self._model.start_chat(history=self._history.to_gemini_history()[:-1])

        # Send the latest user message
        response = chat_session.send_message(user_message)

        # Agentic loop: keep dispatching tools until Gemini returns text
        for _round in range(self.MAX_TOOL_ROUNDS):
            candidate = response.candidates[0]
            parts      = candidate.content.parts

            # Collect all function calls in this response
            fn_calls = [p for p in parts if hasattr(p, "function_call") and p.function_call.name]

            if not fn_calls:
                # No tool calls — extract text and we're done
                text = "".join(
                    p.text for p in parts if hasattr(p, "text") and p.text
                ).strip()
                if not text:
                    text = "(No response from model.)"
                self._history.add_model(text)
                return text, tool_log

            # Dispatch each function call
            fn_results = []
            for part in fn_calls:
                fc   = part.function_call
                name = fc.name
                args = dict(fc.args)   # Struct → plain dict

                logger.info("[agent] tool call: %s(%s)", name, json.dumps(args, default=str))

                fn = TOOL_FUNCTIONS.get(name)
                if fn is None:
                    result = {"error": f"Unknown tool: {name}"}
                else:
                    try:
                        result = fn(**args)
                    except Exception as exc:
                        logger.exception("[agent] tool error: %s", name)
                        result = {"error": str(exc)}

                tool_log.append({"tool": name, "args": args, "result": result})
                fn_results.append({"fn_name": name, "result": result})

            # Send all tool results back to Gemini in one turn
            # Gemini expects a list of Part objects with functionResponse
            from google.generativeai.types import content_types
            tool_response_parts = [
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name   = fr["fn_name"],
                        response = {"result": fr["result"]},
                    )
                )
                for fr in fn_results
            ]
            response = chat_session.send_message(tool_response_parts)

        # If we exhausted MAX_TOOL_ROUNDS without a text response
        fallback = "I couldn't complete your request within the allowed number of tool calls. Please try rephrasing."
        self._history.add_model(fallback)
        return fallback, tool_log

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear conversation history (start fresh)."""
        self._history = ConversationHistory()

    @property
    def message_count(self) -> int:
        return len(self._history.messages)


# ---------------------------------------------------------------------------
# Singleton checker used by the UI
# ---------------------------------------------------------------------------

def is_api_key_configured() -> bool:
    """Return True if the user has set a real API key."""
    key = os.getenv("LLM_API_KEY", "")
    return bool(key) and key != "test-key-placeholder"
