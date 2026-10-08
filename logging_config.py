"""Shared logging configuration for the whole application.

One place to configure logging so the chatbot, the Embedding Manager GUI, and
the ingestion engine all behave consistently. Call ``setup_logging()`` once at
each entrypoint's startup; use ``logging.getLogger(__name__)`` everywhere else.

Design
------
- Standard-library ``logging`` only (no heavy tracing dep yet; Langfuse/OTel are
  a later, deeper layer of observability).
- Console handler by default, so logs still appear in the terminal and in
  ``docker compose logs``.
- Level is env-configurable via ``LOG_LEVEL`` (default INFO).
- Optional file output via ``LOG_FILE`` (useful with a mounted volume in Docker).
- ``setup_logging()`` is idempotent — safe to call on every Streamlit rerun.

Note: the evaluation harness (tests/eval) prints its report to stdout on
purpose and does NOT use this module — report output is not logging.
"""

import logging
import os
import sys

_CONFIGURED = False

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(force=False):
    """Configure the root logger once. Idempotent across Streamlit reruns.

    Env vars:
      LOG_LEVEL  — DEBUG/INFO/WARNING/ERROR/CRITICAL (default INFO)
      LOG_FILE   — if set, also append logs to this path
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return logging.getLogger()

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Clear any pre-existing handlers (e.g. Streamlit/third-party defaults, or a
    # prior rerun) so we don't duplicate log lines.
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    # Console (stderr so it doesn't collide with any stdout report output).
    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Optional file handler.
    log_file = os.getenv("LOG_FILE")
    if log_file:
        try:
            fileh = logging.FileHandler(log_file, encoding="utf-8")
            fileh.setFormatter(formatter)
            root.addHandler(fileh)
        except OSError as ex:
            root.warning("could not open LOG_FILE %s: %s", log_file, ex)

    # Quiet down noisy third-party loggers unless explicitly debugging.
    if level > logging.DEBUG:
        for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "fastembed"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return root


def get_logger(name):
    """Convenience: ensure logging is configured, then return a named logger."""
    setup_logging()
    return logging.getLogger(name)
