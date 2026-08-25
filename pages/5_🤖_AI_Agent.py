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
st.caption("Conversational IPL analytics · Powered by Gemini 2.0 Flash + tool calling")

# ── API key guard ─────────────────────────────────────────────────────────────
if not is_api_key_configured():
    st.warning(
        "### ⚙️ Gemini API Key Required\n\n"
        "The AI Agent needs a Gemini API key to work.\n\n"
        "**Get a free key:** https://aistudio.google.com/apikey\n\n"
        "**Then add it to your `.env` file:**\n"
        "```\n"
        "LLM_API_KEY=AIza...\n"
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
    st.markdown(f"**Messages:** {agent.message_count // 2}")

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
        "This agent has access to **ball-by-ball IPL data** from 2007–2024 "
        "(1,243 matches).\n\n"
        "It **never fabricates stats** — it calls data tools before answering."
    )

# ── Render existing chat history ───────────────────────────────────────────────
chat_messages: list[dict] = st.session_state.get("chat_messages", [])

for msg in chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_log"):
            with st.expander(f"🔧 Tools used ({len(msg['tool_log'])} call{'s' if len(msg['tool_log']) != 1 else ''})", expanded=False):
                for call in msg["tool_log"]:
                    st.markdown(f"**`{call['tool']}`**")
                    col_args, col_res = st.columns(2)
                    with col_args:
                        st.caption("Arguments")
                        st.json(call["args"])
                    with col_res:
                        st.caption("Result")
                        st.json(call["result"])

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
            with st.expander(f"🔧 Tools used ({len(tool_log)} call{'s' if len(tool_log) != 1 else ''})", expanded=False):
                for call in tool_log:
                    st.markdown(f"**`{call['tool']}`**")
                    col_args, col_res = st.columns(2)
                    with col_args:
                        st.caption("Arguments")
                        st.json(call["args"])
                    with col_res:
                        st.caption("Result")
                        st.json(call["result"])

    chat_messages.append({
        "role":     "assistant",
        "content":  reply,
        "tool_log": tool_log,
    })
    st.session_state["chat_messages"] = chat_messages

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
    "CricketIQ AI · Data: IPL 2007–2024 · Model: Gemini 2.0 Flash · "
    "Stats sourced from Cricsheet.org ball-by-ball records"
)
