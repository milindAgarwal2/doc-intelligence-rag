"""Q&A history storage (Postgres).

Stores every answered question + its response + metadata so the admin GUI can
review usage. Written by the SHARED core (chatBOT_engine.generate_answer), so
BOTH the Streamlit chatbot and the widget API log automatically.

Design principles
-----------------
- **Fail-safe.** Logging must NEVER break answering. Every public function
  swallows DB errors (logs a warning) and degrades gracefully: log_qa() becomes
  a no-op, read functions return empty results. A missing/unreachable DB simply
  means "no history", not a broken chatbot.
- **Lazy + pooled.** A single connection pool is created on first use. If the DB
  is unreachable the pool creation fails once, is remembered, and we stop trying
  for the process lifetime (so we don't hammer a dead DB on every question).
- **Secrets from env.** Connection params (incl. password) come from the
  environment (.env). The on/off toggle + retention live in config.json.
- **Schema auto-created.** ensure_schema() runs once; no migration tool needed.

Env vars (see .env.example):
  POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, DB_HOST, DB_PORT
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any

logger = logging.getLogger("db")

# Lazy singletons guarded by a lock (api_server/streamlit may call concurrently).
_pool = None
_pool_init_failed = False
_schema_ready = False
_lock = threading.Lock()

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS qa_log (
    id             BIGSERIAL PRIMARY KEY,
    asked_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    source         TEXT        NOT NULL DEFAULT 'chatbot',
    question       TEXT        NOT NULL,
    answer         TEXT        NOT NULL,
    provider       TEXT,
    best_score     REAL,
    low_confidence BOOLEAN,
    refusal        BOOLEAN,
    sources        JSONB,
    latency_ms     INTEGER
);
CREATE INDEX IF NOT EXISTS qa_log_asked_at_idx ON qa_log (asked_at DESC);
CREATE INDEX IF NOT EXISTS qa_log_source_idx   ON qa_log (source);
"""


def _conninfo() -> str:
    """Build a libpq conninfo string from the environment."""
    host = os.getenv("DB_HOST", "db")
    port = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("POSTGRES_DB", "techdoc")
    user = os.getenv("POSTGRES_USER", "techdoc")
    password = os.getenv("POSTGRES_PASSWORD", "")
    return (
        f"host={host} port={port} dbname={dbname} "
        f"user={user} password={password} connect_timeout=5"
    )


def _get_pool():
    """Return a connection pool, creating it once. Returns None if the DB is
    unreachable (remembered, so we don't retry forever)."""
    global _pool, _pool_init_failed
    if _pool is not None:
        return _pool
    if _pool_init_failed:
        return None
    with _lock:
        if _pool is not None:
            return _pool
        if _pool_init_failed:
            return None
        try:
            from psycopg_pool import ConnectionPool

            pool = ConnectionPool(
                conninfo=_conninfo(),
                min_size=1,
                max_size=4,
                timeout=3,
                # Verify connectivity up front (raises if the DB is unreachable),
                # so a dead DB is detected ONCE here and remembered via
                # _pool_init_failed — rather than paying the connect timeout on
                # every answer. open=False + explicit open(wait) gives us that.
                open=False,
            )
            pool.open(wait=True, timeout=5)
            _pool = pool
            logger.info("qa-log: Postgres connection pool ready")
            return _pool
        except Exception as ex:  # unreachable DB, bad creds, driver missing…
            _pool_init_failed = True
            logger.warning("qa-log: DB unavailable, logging disabled (%s)", ex)
            return None


def _ensure_schema(conn) -> bool:
    """Create the table/indexes once per process. Returns True if ready."""
    global _schema_ready
    if _schema_ready:
        return True
    with _lock:
        if _schema_ready:
            return True
        try:
            with conn.cursor() as cur:
                cur.execute(_CREATE_SQL)
            conn.commit()
            _schema_ready = True
            return True
        except Exception as ex:
            logger.warning("qa-log: schema init failed (%s)", ex)
            return False


def is_available() -> bool:
    """True if the DB is reachable (used by the GUI to show a clear status)."""
    pool = _get_pool()
    if pool is None:
        return False
    try:
        with pool.connection() as conn:
            return _ensure_schema(conn)
    except Exception as ex:
        logger.warning("qa-log: availability check failed (%s)", ex)
        return False


def log_qa(
    *,
    question: str,
    answer: str,
    source: str = "chatbot",
    provider: str | None = None,
    best_score: float | None = None,
    low_confidence: bool | None = None,
    refusal: bool | None = None,
    sources: list[str] | None = None,
    latency_ms: int | None = None,
) -> bool:
    """Insert one Q&A row. FAIL-SAFE: returns False on any error, never raises.

    Called from generate_answer() for both the chatbot and the widget. A False
    return (DB down, logging misconfigured) must not affect the answer the user
    already received.
    """
    if not question or not answer:
        return False
    pool = _get_pool()
    if pool is None:
        return False
    try:
        with pool.connection() as conn:
            if not _ensure_schema(conn):
                return False
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO qa_log
                      (source, question, answer, provider, best_score,
                       low_confidence, refusal, sources, latency_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        source,
                        question,
                        answer,
                        provider,
                        best_score,
                        low_confidence,
                        refusal,
                        json.dumps(sources or []),
                        latency_ms,
                    ),
                )
            conn.commit()
        return True
    except Exception as ex:
        logger.warning("qa-log: insert failed (%s)", ex)
        return False


def fetch_qa(
    *,
    limit: int = 200,
    offset: int = 0,
    source: str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """Return recent Q&A rows (newest first) as a list of dicts for the GUI.

    Optional filters: ``source`` ('chatbot'|'widget') and a case-insensitive
    ``search`` substring over the question text. FAIL-SAFE: returns [] on error.
    """
    pool = _get_pool()
    if pool is None:
        return []
    where = []
    params: list[Any] = []
    if source in ("chatbot", "widget"):
        where.append("source = %s")
        params.append(source)
    if search:
        where.append("question ILIKE %s")
        params.append(f"%{search}%")
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    sql = (
        "SELECT id, asked_at, source, question, answer, provider, best_score, "
        "low_confidence, refusal, sources, latency_ms "
        f"FROM qa_log {clause} ORDER BY asked_at DESC LIMIT %s OFFSET %s"
    )
    params.extend([max(1, min(limit, 2000)), max(0, offset)])
    try:
        with pool.connection() as conn:
            if not _ensure_schema(conn):
                return []
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [c.name for c in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as ex:
        logger.warning("qa-log: fetch failed (%s)", ex)
        return []


def stats() -> dict[str, Any]:
    """Return summary stats for the GUI header. FAIL-SAFE: zeros on error."""
    empty = {"total": 0, "chatbot": 0, "widget": 0, "refusals": 0,
             "low_conf": 0, "first": None, "last": None}
    pool = _get_pool()
    if pool is None:
        return empty
    try:
        with pool.connection() as conn:
            if not _ensure_schema(conn):
                return empty
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                      count(*),
                      count(*) FILTER (WHERE source = 'chatbot'),
                      count(*) FILTER (WHERE source = 'widget'),
                      count(*) FILTER (WHERE refusal),
                      count(*) FILTER (WHERE low_confidence),
                      min(asked_at), max(asked_at)
                    FROM qa_log
                    """
                )
                r = cur.fetchone()
        return {
            "total": r[0] or 0,
            "chatbot": r[1] or 0,
            "widget": r[2] or 0,
            "refusals": r[3] or 0,
            "low_conf": r[4] or 0,
            "first": r[5],
            "last": r[6],
        }
    except Exception as ex:
        logger.warning("qa-log: stats failed (%s)", ex)
        return empty


def purge_older_than(days: int) -> int:
    """Delete rows older than ``days`` (0 = keep forever -> no-op). Returns the
    number of rows deleted. FAIL-SAFE: returns 0 on error."""
    if not days or days <= 0:
        return 0
    pool = _get_pool()
    if pool is None:
        return 0
    try:
        with pool.connection() as conn:
            if not _ensure_schema(conn):
                return 0
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM qa_log "
                    "WHERE asked_at < now() - make_interval(days => %s)",
                    (days,),
                )
                deleted = cur.rowcount
            conn.commit()
        if deleted:
            logger.info("qa-log: purged %d row(s) older than %d days", deleted, days)
        return deleted or 0
    except Exception as ex:
        logger.warning("qa-log: purge failed (%s)", ex)
        return 0


def clear_all() -> bool:
    """Delete ALL Q&A rows (admin action). FAIL-SAFE: returns False on error."""
    pool = _get_pool()
    if pool is None:
        return False
    try:
        with pool.connection() as conn:
            if not _ensure_schema(conn):
                return False
            with conn.cursor() as cur:
                cur.execute("TRUNCATE qa_log RESTART IDENTITY")
            conn.commit()
        logger.info("qa-log: all rows cleared")
        return True
    except Exception as ex:
        logger.warning("qa-log: clear failed (%s)", ex)
        return False
