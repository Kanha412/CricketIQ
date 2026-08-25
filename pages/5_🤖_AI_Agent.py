"""
CricketIQ — AI Agent Chat Page
================================
Conversational IPL analytics powered by Gemini 2.0 Flash + function calling.

Features:
  • Multi-turn chat with full history
  • Tool call transparency (shows which tools were called)
  • Graceful fallback when API key is not configured
  • Session-persistent agent (no reload on re-render)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import app.cloud  # noqa: F401 — Streamlit secrets → env vars bridge (cloud deployment)

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # local dev — cloud uses st.secrets via app.cloud above  # must be before agent imports so LLM_API_KEY is in env

from app.utils import cached_db_ok
from app.components import require_db, section
from src.ai.agent import CricketIQAgent, is_api_key_configured

# ── DB check ──────────────────────────────────────────────────────────────────
require_db(cached_db_ok())

# ── Page config ───────────────────────────────────────────────────────────────
st.title("🤖 CricketIQ AI Agent")
st.caption("Conversational IPL analytics · Powered by CricketIQ AI + tool calling")

# ── API key guard ─────────────────────────────────────────────────────────────
if not is_api_key_configured():
    st.warning(
        "### ⚙️ AI API Key Required\n\n"
        "The AI Agent needs an API key to work.\n\n"
        "**Add it to your `.env` file:**\n"
        "```\n"
        "LLM_API_KEY=your-key-here\n"
        "```\n\n"
        "Restart the Streamlit app after saving the key."
    )
    st.info(
        "💡 While waiting, you can explore the other pages:\n"
        "- **Player Intelligence** — batting & bowling deep-dives\n"
        "- **Matchup Engine** — batter vs bowler stats\n"
        "- **Team Intelligence** — head-to-head records\n"
        "- **Win Probability** — ML win predictor + simulator"
    )
    st.stop()

# ── Agent initialisation (persisted in session_state) ─────────────────────────
if "cricketiq_agent" not in st.session_state:
    try:
        st.session_state["cricketiq_agent"] = CricketIQAgent.create()
        st.session_state["chat_messages"]   = []   # [{role, content, tool_log}]
    except RuntimeError as exc:
        if "placeholder_key" in str(exc):
            st.error("API key is still the placeholder. Update LLM_API_KEY in .env")
        elif "genai_not_installed" in str(exc):
            st.error(
                "google-generativeai package not installed.\n"
                "Run: `uv add google-generativeai`"
            )
        else:
            st.error(f"Failed to initialise agent: {exc}")
        st.stop()

agent: CricketIQAgent = st.session_state["cricketiq_agent"]

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 AI Agent")
    n_exchanges = len(st.session_state.get("chat_messages", [])) // 2
    st.markdown(f"**Messages:** {n_exchanges}")

    if st.button("🔄 New conversation", use_container_width=True):
        agent.reset()
        st.session_state["chat_messages"] = []
        st.rerun()

    st.divider()
    st.markdown("### 💡 Example questions")
    examples = [
        "What are Virat Kohli's career batting stats?",
        "Compare Bumrah vs Rohit head-to-head",
        "How does Kohli perform in the death overs?",
        "What is CSK's record vs MI?",
        "Should I bat first at Wankhede?",
        "If a team needs 80 runs in 9 overs with 3 wickets down, who wins?",
        "Simulate a full T20 innings",
        "Who bowls best in the powerplay?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state["pending_input"] = ex

    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        "This agent has access to **ball-by-ball IPL data** from 2008–2024 "
        "(1,243 matches).\n\n"
        "It **never fabricates stats** — it calls data tools before answering."
    )

# ── Tool log → human readable ─────────────────────────────────────────────────
_TOOL_LABELS = {
    "get_batting_stats":      ("📊", "Batting stats"),
    "get_bowling_stats":      ("🎳", "Bowling stats"),
    "get_matchup":            ("⚔️", "Batter vs bowler matchup"),
    "get_team_head_to_head":  ("🏆", "Team head-to-head"),
    "predict_win_probability":("🎯", "Win probability"),
    "simulate_innings_score": ("🎲", "Innings simulation"),
    "search_player_name":     ("🔍", "Player name search"),
    "get_venue_stats":        ("🏟️", "Venue stats"),
}

def _tool_summary(call: dict) -> str | None:
    """
    Convert one tool call dict into a single plain-English line.
    Returns None for failed lookups — callers skip those.
    """
    result = call.get("result", {})
    if "error" in result:
        return None   # internal retry — don't show to user

    icon, label = _TOOL_LABELS.get(call["tool"], ("🔧", call["tool"]))
    args = call.get("args", {})

    subject = (
        args.get("player_name")
        or args.get("query")
        or args.get("batter_name")
        or args.get("team_a")
        or args.get("venue_name")
        or ""
    )
    suffix = f" for **{subject}**" if subject else ""
    return f"{icon} {label}{suffix}"


def _render_tool_log(tool_log: list[dict]) -> None:
    """Render only successful lookups in a compact collapsed expander."""
    if not tool_log:
        return
    lines   = [_tool_summary(c) for c in tool_log]
    visible = [l for l in lines if l is not None]
    if not visible:
        return
    n_retried = len(lines) - len(visible)
    retry_note = f" · {n_retried} name lookup{'s' if n_retried != 1 else ''} resolved automatically" if n_retried else ""
    label = f"🔍 Data verified from {len(visible)} source{'s' if len(visible) != 1 else ''}{retry_note}"
    with st.expander(label, expanded=False):
        for line in visible:
            st.markdown(line)


# ── Render existing chat history ───────────────────────────────────────────────
chat_messages: list[dict] = st.session_state.get("chat_messages", [])

for msg in chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_log"):
            _render_tool_log(msg["tool_log"])

# ── Handle pre-filled input from sidebar buttons ───────────────────────────────
pending = st.session_state.pop("pending_input", None)

# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me anything about IPL stats, players, or matches…")

# Use pending sidebar example if no manual input this turn
if pending and not user_input:
    user_input = pending

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    chat_messages.append({"role": "user", "content": user_input, "tool_log": []})

    # Run agent with a spinner
    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):
            try:
                reply, tool_log = agent.chat(user_input)
            except Exception as exc:
                reply    = f"⚠️ Error: {exc}\n\nTry rephrasing or check the logs."
                tool_log = []

        st.markdown(reply)

        if tool_log:
            _render_tool_log(tool_log)

    chat_messages.append({
        "role":     "assistant",
        "content":  reply,
        "tool_log": tool_log,
    })
    st.session_state["chat_messages"] = chat_messages
    st.rerun()  # re-render so sidebar counter updates immediately

# ── Empty state prompt ─────────────────────────────────────────────────────────
if not chat_messages:
    st.markdown("---")
    section("What can I help you with?", "💬")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Player stats**\nAsset batting avg, SR, dismissals, phases")
    with c2:
        st.info("**Matchups**\nBatter vs bowler dominance analysis")
    with c3:
        st.info("**Match state**\nWin probability for any chase scenario")

st.divider()
st.caption(
    "CricketIQ AI ·  Data: IPL 2008–2024  · "
    "Stats sourced from Cricsheet.org ball-by-ball records"
)
