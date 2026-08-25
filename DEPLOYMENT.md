# CricketIQ — Deployment Guide

Deploy CricketIQ to **Streamlit Community Cloud** (free) backed by **Neon** (free serverless PostgreSQL).

---

## Overview

| Layer | Local Dev | Production |
|---|---|---|
| Database | PostgreSQL 17 on `localhost` | Neon (serverless PostgreSQL, free tier) |
| App server | `streamlit run streamlit_app.py` | Streamlit Community Cloud (free tier) |
| ML models | `models/*.joblib` (local) | Committed to git → loaded by Streamlit Cloud |
| Secrets | `.env` file | Streamlit Cloud secrets UI |

---

## Step 1 — Create a Neon database

1. Sign up at **https://console.neon.tech** (free, no credit card)
2. Create a new project → name it `cricketiq`
3. Neon auto-creates a default database. Copy the **Connection string**:
   ```
   postgresql://kanha:<password>@ep-xxx-yyy.region.aws.neon.tech/cricketiq?sslmode=require
   ```
4. Save it — you'll need it in Step 2.

---

## Step 2 — Migrate data to Neon

### Option A — pg_dump (recommended, fastest)

Add your Neon URL to `.env`:
```bash
NEON_DATABASE_URL=postgresql://user:pass@ep-xxx.region.aws.neon.tech/cricketiq?sslmode=require
```

Run the migration script:
```bash
# First: run Alembic migrations against Neon to create tables
DATABASE_URL=$NEON_DATABASE_URL alembic upgrade head

# Then: migrate data
python scripts/migrate_to_neon.py --dry-run   # preview commands
python scripts/migrate_to_neon.py              # actually migrate
```

### Option B — Re-ingest from source (if pg_dump unavailable)

```bash
# Point all scripts at Neon
set DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/cricketiq?sslmode=require

alembic upgrade head
python scripts/ingest_matches.py
python scripts/compute_distributions.py
python scripts/train_win_model.py
```

> **Note:** Re-ingestion takes ~10 minutes for 295,732 deliveries.

### Verify Neon data

```bash
# Test connection
DATABASE_URL=$NEON_DATABASE_URL python -c "
from src.db.connection import test_connection
print('OK' if test_connection() else 'FAIL')
"
```

---

## Step 3 — Push to GitHub

1. Create a new **public** repository on GitHub (e.g. `cricketiq`)
   - Do NOT initialise with README (our commit is the initial commit)

2. Add the remote and push:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/cricketiq.git
   git push -u origin main
   ```

> **ML models** are committed (`models/*.joblib`) — Streamlit Cloud will use them without retraining.

---

## Step 4 — Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** → sign in with GitHub

2. Click **New app** → select your repository:
   - **Repository:** `YOUR_USERNAME/cricketiq`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`

3. Click **Advanced settings → Secrets** and paste:
   ```toml
   DATABASE_URL = "postgresql://user:pass@ep-xxx.region.aws.neon.tech/cricketiq?sslmode=require"
   LLM_API_KEY  = "AIzaSy..."
   ENVIRONMENT  = "production"
   LOG_LEVEL    = "WARNING"
   ```

4. Click **Deploy** — Streamlit installs dependencies from `requirements.txt` + `packages.txt`.

> First deploy takes ~3 minutes. Subsequent deploys are faster (cached layers).

---

## Step 5 — Verify the deployment

After deployment completes, open the app URL and check:

| Page | What to verify |
|---|---|
| Home | Shows 1,243 matches, 295,732 deliveries metrics |
| Player Intelligence | Kohli batting stats load correctly |
| Matchup Engine | Kohli vs Bumrah dominance gauge renders |
| Win Probability | Model loads, gauge shows probability |
| AI Agent | Shows "API key required" or chat interface |

---

## Troubleshooting

### App fails to start — `DATABASE_URL not found`
- Check Streamlit Cloud secrets — make sure `DATABASE_URL` is set (no extra spaces)
- The secrets bridge (`app/cloud.py`) copies secrets to env vars on startup

### DB connection refused
- Neon free-tier projects pause after 5 minutes of inactivity
- First request after pause takes ~2s to cold-start — this is normal
- `NullPool` is used for Neon so stale connections don't cause errors

### Win Probability page: "Model not trained"
- Check that `models/win_probability_model.joblib` is committed:
  ```bash
  git ls-files models/
  ```
  Should show both `.joblib` files.

### AI Agent — tool call errors
- Add `LLM_API_KEY` to Streamlit secrets
- Get a free Gemini key at https://aistudio.google.com/apikey

### ImportError on a page
- Check `requirements.txt` is up to date: `uv pip compile pyproject.toml -o requirements.txt`
- Re-push to trigger a fresh Streamlit build

---

## Maintenance

### Refresh Neon data (new IPL season)
```bash
# Download latest Cricsheet data
python scripts/download_cricsheet.py

# Ingest new matches only (script checks for existing matches)
DATABASE_URL=$NEON_DATABASE_URL python scripts/ingest_matches.py

# Retrain model (optional — only if data grew significantly)
DATABASE_URL=$NEON_DATABASE_URL python scripts/train_win_model.py
DATABASE_URL=$NEON_DATABASE_URL python scripts/compute_distributions.py

# Commit updated models
git add models/
git commit -m "chore: retrain models with IPL 2025 data"
git push
```

### Monitor Neon usage
- Neon free tier: 0.5 GB storage, 3 GB transfer/month
- Check usage at https://console.neon.tech
- 295K deliveries ≈ 35 MB on disk — well within free tier

---

## Architecture Diagram

```
[User Browser]
      │ HTTPS
      ▼
[Streamlit Community Cloud]
  streamlit_app.py
  pages/*.py
  app/cloud.py ──── reads st.secrets → sets os.environ
      │
      ├── src/analytics/   (pandas, numpy)
      ├── src/ml/          (joblib loads models/*.joblib)
      ├── src/simulation/  (numpy Monte Carlo)
      ├── src/ai/          (Gemini 2.0 Flash API)
      │
      └── src/db/ ─────────────────────────────────────────────►  [Neon PostgreSQL]
            NullPool (serverless-safe)                             SSL required
            pool_pre_ping=True
```
