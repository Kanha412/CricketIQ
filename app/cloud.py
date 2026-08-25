"""
CricketIQ — Cloud / Streamlit Secrets Bridge
==============================================
Import this at the very top of every page file (before any src.* imports)
when deploying to Streamlit Community Cloud.

WHY THIS EXISTS:
    Streamlit Community Cloud stores secrets in its own vault (the cloud UI).
    These are exposed as st.secrets["KEY"], NOT as os.environ["KEY"].
    Pydantic-settings reads from os.environ, so without this bridge the app
    can't find DATABASE_URL or LLM_API_KEY on the cloud.

    This module copies Streamlit secrets → os.environ once per process.
    On local dev the .env file is already loaded by pydantic-settings, so
    st.secrets is empty/irrelevant — the copy is a no-op.

USAGE (first line of every page, before other imports):
    import app.cloud  # noqa: F401  — secrets bridge
"""

import os

_SECRETS_KEYS = [
    "DATABASE_URL",
    "LLM_API_KEY",
    "ENVIRONMENT",
    "LOG_LEVEL",
    "LLM_PROVIDER",
    "LLM_MODEL",
]

try:
    import streamlit as st
    for _key in _SECRETS_KEYS:
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass   # Not in Streamlit context (tests, scripts) — ignore
