# CricketIQ 🏏

> **AI-powered cricket analytics and tactical intelligence platform**

CricketIQ transforms raw ball-by-ball cricket data from [Cricsheet](https://cricsheet.org) into
deep analytical insights — using real statistical calculations, machine learning, and a grounded
AI analyst that calls Python tools before generating any explanation.

---

## Live Demo

🚀 **[Launch CricketIQ](https://cricketiq.streamlit.app)** *(coming soon)*

---

## What CricketIQ Does

- **Match Autopsy** — Automated phase-by-phase investigation of any completed match
- **Player Intelligence** — Multi-dimensional player profiles with rolling form metrics
- **Batter vs Bowler Matchup Engine** — Head-to-head statistics with confidence indicators
- **Win Probability Tracker** — Ball-by-ball win probability from a trained ML model
- **What-If Simulator** — Monte Carlo simulation of alternate tactical decisions
- **AI Cricket Analyst** — Agent that dispatches Python tools first, then narrates the findings

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Source | Cricsheet (ball-by-ball JSON) |
| Analytics | Python · Pandas · NumPy |
| Machine Learning | scikit-learn (Logistic Regression, XGBoost) |
| Simulation | NumPy Monte Carlo |
| Visualization | Matplotlib |
| Frontend | Streamlit |
| Database | PostgreSQL (Neon free tier) |
| AI / LLM | Gemini 2.0 Flash (free tier) |
| Embeddings | sentence-transformers (local, no API) |
| Deployment | Streamlit Community Cloud · GitHub Actions |

**Total running cost: ₹0/month**

---

## Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- PostgreSQL running locally (or a Neon free account)
- `uv` package manager: `pip install uv`

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/cricketiq.git
cd cricketiq

# 2. Install dependencies
uv venv
uv pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env with your DATABASE_URL and LLM_API_KEY

# 4. Set up the database
alembic upgrade head

# 5. Download IPL sample data from Cricsheet
python scripts/download_cricsheet.py --competitions ipl

# 6. Ingest data into PostgreSQL
python scripts/ingest_matches.py

# 7. Launch the app
streamlit run app/Home.py
```

---

## Project Structure

```
cricketiq/
├── app/               # Streamlit pages and UI components
├── src/
│   ├── analytics/     # Pandas/NumPy analysis functions
│   ├── models/        # ML models (win probability, player rating)
│   ├── simulation/    # Monte Carlo what-if engine
│   ├── ai/            # AI agent + LLM provider interfaces
│   ├── visualization/ # Matplotlib chart functions
│   └── db/            # Database queries and connection
├── data/samples/      # Sample match JSONs for testing
├── scripts/           # Data download, ingestion, training
└── tests/             # Unit and integration tests
```

---

## Data Source

All cricket data is from [Cricsheet.org](https://cricsheet.org) — free, openly licensed
ball-by-ball data covering IPL, T20 Internationals, ODIs, Tests, and more.

No live data scraping. Data is downloaded as ZIP files and ingested into PostgreSQL.

---

## Architecture

```
Cricsheet JSON → Parse → Validate → PostgreSQL
                                        ↓
                               Analytics Engine
                               (Pandas + NumPy)
                                        ↓
                              ┌─────────┴──────────┐
                              ↓                    ↓
                         Matplotlib           ML Models
                           Charts          Win Probability
                              ↓                    ↓
                          AI Agent (tool-calling)
                              ↓
                          Streamlit UI
```

---

## Development Roadmap

- [x] Phase 1: Project foundation
- [ ] Phase 2: Cricsheet data ingestion
- [ ] Phase 3: Database schema + ingestion pipeline
- [ ] Phase 4: Core analytics engine
- [ ] Phase 5: Visualization layer
- [ ] Phase 6: Streamlit UI (MVP pages)
- [ ] Phase 7: Win probability ML model
- [ ] Phase 8: AI analyst + Match Autopsy
- [ ] Phase 9: What-If simulator
- [ ] Phase 10: RAG knowledge base (V2)
- [ ] Phase 11: FastAPI backend (V2)
- [ ] Phase 12: Docker (V2)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Data from Cricsheet is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
