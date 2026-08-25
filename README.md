# 🏏 CricketIQ

> **AI-powered IPL analytics & tactical intelligence platform**

CricketIQ turns 1,243 IPL matches and 295,000+ ball-by-ball deliveries into interactive analytics — with a conversational AI analyst that queries real data before answering, never guessing.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cricketiq.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PostgreSQL](https://img.shields.io/badge/Database-Neon%20PostgreSQL-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Live Demo

**[cricketiq.streamlit.app](https://cricketiq.streamlit.app)** *(deploying soon)*

---

## ✨ Features

| Page | What it does |
|------|-------------|
| 🏏 **Player Intelligence** | Career stats, phase breakdown (powerplay/middle/death), season trends, dismissal analysis, venue performance, scoring map by over |
| ⚔️ **Matchup Engine** | Batter vs bowler head-to-head — balls faced, dismissals, dominance index, phase split, over-by-over breakdown |
| 🏆 **Team Intelligence** | Win rates for all 19 teams, head-to-head records, venue analysis (bat-first %), toss impact, phase scoring patterns |
| 🎯 **Win Probability** | ML model (XGBoost) trained on all IPL chases — input any match state and get win % + Monte Carlo score simulation |
| 🤖 **AI Agent** | Conversational IPL analyst — asks the database before answering, never fabricates numbers. Powered by tool-calling LLM |

---

## 📊 Data

- **Source:** [Cricsheet.org](https://cricsheet.org) — free, openly licensed ball-by-ball JSON
- **Coverage:** IPL 2008–2026 · 1,243 matches · 295,732 deliveries · 964 players · 19 teams · 37 venues
- **No live scraping** — downloaded as ZIP, bulk-ingested into PostgreSQL (Neon)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Package manager | `uv` |
| Database | PostgreSQL via [Neon](https://neon.tech) (serverless, free tier) |
| Analytics | Pandas · NumPy |
| Machine Learning | scikit-learn · XGBoost |
| Simulation | NumPy Monte Carlo (10,000 innings) |
| Visualisation | Matplotlib |
| Frontend | Streamlit |
| AI / LLM | Gemini (tool-calling, free tier) |
| Deployment | Streamlit Community Cloud |

**Total running cost: ₹0 / month**

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- A free [Neon](https://neon.tech) PostgreSQL account
- A free [Google AI Studio](https://aistudio.google.com) API key (for the AI Agent)

### 1 — Clone & install

```bash
git clone https://github.com/Kanha412/CricketIQ.git
cd CricketIQ
pip install uv
uv venv && uv sync
```

### 2 — Configure environment

```bash
cp .streamlit/secrets.toml.example .env
# Fill in DATABASE_URL and LLM_API_KEY
```

`.env` format:
```
DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
LLM_API_KEY=your-key-here
LLM_MODEL=gemini-3.6-flash
```

### 3 — Set up database

```bash
uv run alembic upgrade head
```

### 4 — Download & ingest IPL data

```bash
# Download IPL JSON from https://cricsheet.org/downloads/ → extract to data/raw/ipl/
uv run python scripts/fast_ingest_neon.py
```

### 5 — Run

```bash
uv run streamlit run streamlit_app.py
```

---

## 🗂️ Project Structure

```
CricketIQ/
├── streamlit_app.py          # Home page (entry point)
├── pages/
│   ├── 1_🏏_Player_Intelligence.py
│   ├── 2_⚔️_Matchup_Engine.py
│   ├── 3_🏆_Team_Intelligence.py
│   ├── 4_🎯_Win_Probability.py
│   └── 5_🤖_AI_Agent.py
├── app/
│   ├── components.py         # Reusable UI components
│   ├── utils.py              # Cached DB wrappers
│   └── cloud.py              # Streamlit secrets bridge
├── src/
│   ├── analytics/            # batting.py, bowling.py, matchup.py, team.py
│   ├── ai/                   # agent.py, tools.py, prompts.py
│   ├── db/                   # connection.py, queries.py, schema.py
│   ├── models/               # win_probability.py, simulator.py
│   └── visualization/        # batting_charts.py, bowling_charts.py, ...
├── scripts/
│   ├── fast_ingest_neon.py   # Bulk ingest (psycopg2 execute_values)
│   └── normalize_venues.py   # One-time venue name deduplication
├── tests/                    # 178 unit tests
├── models/                   # Trained .joblib model files
└── data/samples/             # Sample JSONs for testing
```

---

## 🤖 AI Agent Design

The AI Agent uses a **tool-calling pattern** — the LLM never calculates anything itself:

```
User question
     ↓
 LLM decides which tool to call
     ↓
 Python function queries DB → runs analytics → returns JSON dict
     ↓
 LLM reads the real numbers and narrates the answer
```

**8 tools available:** `get_batting_stats`, `get_bowling_stats`, `get_matchup`,
`get_team_head_to_head`, `predict_win_probability`, `simulate_innings_score`,
`search_player_name`, `get_venue_stats`

The agent auto-resolves player names (e.g. "Jasprit Bumrah" → "JJ Bumrah" from the DB) without user intervention.

---

## 🚢 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for full Neon + Streamlit Cloud instructions.

**Short version:**
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select repo
3. Add secrets in Streamlit Cloud settings (DATABASE_URL, LLM_API_KEY, LLM_MODEL)
4. Deploy ✓

---

## 📝 License

MIT License — see [LICENSE](LICENSE).

Data from [Cricsheet.org](https://cricsheet.org) is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
