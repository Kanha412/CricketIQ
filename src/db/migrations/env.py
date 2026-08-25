"""
Alembic Migration Environment
================================
This file configures how Alembic connects to the database and
discovers schema changes.

KEY DESIGN: We import our MetaData (from src/db/tables.py) so Alembic
can compare the current DB state against the defined schema and generate
'--autogenerate' migrations in the future.

We read DATABASE_URL from src/config.py (which reads from .env),
NOT from alembic.ini. This keeps secrets out of config files.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.config import settings
from src.db.tables import metadata   # Our schema — Alembic reads this

# ── Alembic Config ────────────────────────────────────────────────────────────
config = context.config

# Set up logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override the sqlalchemy.url with the real URL from our settings
config.set_main_option("sqlalchemy.url", settings.database_url)

# This is the MetaData Alembic uses for --autogenerate
target_metadata = metadata


# ── Migration Runners ─────────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    Generates SQL script without connecting to the database.
    Useful for reviewing SQL before applying.
    Usage: alembic upgrade head --sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    Connects to the database and applies migrations directly.
    Normal usage: alembic upgrade head
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,    # No connection pooling for migration runs
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,      # Detect column type changes
        )
        with context.begin_transaction():
            context.run_migrations()


# ── Entry Point ───────────────────────────────────────────────────────────────
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
