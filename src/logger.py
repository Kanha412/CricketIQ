"""
CricketIQ — Logging Setup
==========================
Centralised logging initialisation.

Usage in any module:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing match %s", match_id)
    logger.error("Failed to parse delivery: %s", error, exc_info=True)
"""

import logging
import logging.config
import logging.handlers
import sys
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "logging_config.yaml"
_LOGS_DIR = Path(__file__).parent.parent / "logs"
_initialized = False


def _ensure_logs_dir() -> None:
    """Create the logs/ directory if it doesn't exist."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(level: str = "INFO") -> None:
    """
    Load logging configuration from YAML and apply it.

    Called once at application startup (in Home.py or scripts/__init__.py).
    Safe to call multiple times — subsequent calls are ignored.

    Args:
        level: Override log level (e.g. from settings.log_level).
               Applied to the root 'src' logger.
    """
    global _initialized
    if _initialized:
        return

    _ensure_logs_dir()

    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Override level from settings if provided
        if level and "loggers" in config and "src" in config["loggers"]:
            config["loggers"]["src"]["level"] = level.upper()

        logging.config.dictConfig(config)
    else:
        # Fallback: basic config if YAML not found
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
        )
        logging.warning("logging_config.yaml not found — using basic config")

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.

    Always call this with __name__ so log messages show the module path.
    Example: get_logger(__name__) in src/analytics/match.py
             → logger name becomes 'src.analytics.match'

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A configured Logger instance.
    """
    if not _initialized:
        # Auto-initialize with default settings if setup_logging wasn't called
        setup_logging()
    return logging.getLogger(name)
