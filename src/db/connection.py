"""
CricketIQ — Database Connection
=================================
SQLAlchemy engine and connection utilities.

DESIGN:
    - Single engine per application (created once at startup)
    - Connection pool for efficient reuse across Streamlit reruns
    - pool_pre_ping=True: validates connections before use (handles stale connections)
    - Context manager pattern: auto-commits or rolls back on error

USAGE:
    # For read queries (no transaction needed):
    from src.db.connection import get_connection
    with get_connection() as conn:
        result = conn.execute(text("SELECT ..."))

    # For write operations (auto-commits on exit, rolls back on exception):
    from src.db.connection import get_transaction
    with get_transaction() as conn:
        conn.execute(insert_stmt)
        # commits when block exits cleanly
        # rolls back if exception raised
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

_engine: Engine | None = None


def _is_neon_url(url: str) -> bool:
    """
    Detect Neon PostgreSQL URLs.
    Neon = serverless PostgreSQL (neon.tech) — uses a pooling proxy.
    Standard connection pooling fights with the proxy; NullPool is safer.
    """
    return "neon.tech" in url or ".neon.tech" in url


def get_engine() -> Engine:
    """
    Return the singleton SQLAlchemy engine.
    Creates it on first call; returns cached instance on subsequent calls.

    LOCAL (standard PostgreSQL):
        pool_size=5, max_overflow=10 — persistent pool, fast re-use.

    NEON (serverless PostgreSQL):
        NullPool — create a new connection per request and close immediately.
        Neon's proxy handles the actual pooling on their side.
        pool_pre_ping is still useful for catching stale proxy connections.
    """
    global _engine
    if _engine is None:
        db_url = settings.database_url
        logger.info("Creating database engine: %s", _masked_url(db_url))

        if _is_neon_url(db_url):
            logger.info("Neon serverless detected — using NullPool")
            _engine = create_engine(
                db_url,
                poolclass=NullPool,
                pool_pre_ping=True,
                echo=False,
                connect_args={"sslmode": "require"},
            )
        else:
            _engine = create_engine(
                db_url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,    # Set to True to log every SQL query (very verbose)
            )
        # Verify connection on startup
        try:
            with _engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection verified successfully.")
        except OperationalError as e:
            logger.error("Database connection failed: %s", e)
            raise
    return _engine


@contextmanager
def get_connection() -> Generator:
    """
    Context manager for read-only database queries.
    Does NOT auto-commit. Use get_transaction() for writes.

    Usage:
        with get_connection() as conn:
            rows = conn.execute(text("SELECT id FROM matches LIMIT 10")).fetchall()
    """
    engine = get_engine()
    with engine.connect() as conn:
        yield conn


@contextmanager
def get_transaction() -> Generator:
    """
    Context manager for write operations.
    Auto-commits when block exits cleanly.
    Auto-rolls back if an exception is raised.

    Usage:
        with get_transaction() as conn:
            conn.execute(matches_table.insert().values(...))
            conn.execute(innings_table.insert().values(...))
            # Both committed atomically, or both rolled back on error
    """
    engine = get_engine()
    with engine.begin() as conn:    # engine.begin() = auto-commit/rollback
        yield conn


def test_connection() -> bool:
    """
    Simple connectivity test. Returns True if database is reachable.
    Used in health checks and startup validation.
    """
    try:
        with get_connection() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error("Connection test failed: %s", e)
        return False


def _masked_url(url: str) -> str:
    """Return the DB URL with password masked for safe logging."""
    if "@" in url and "://" in url:
        scheme_user, rest = url.split("@", 1)
        if ":" in scheme_user:
            scheme_user = scheme_user.rsplit(":", 1)[0] + ":***"
        return f"{scheme_user}@{rest}"
    return url
