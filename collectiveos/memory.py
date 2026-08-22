"""
Memory store — Postgres + pgvector with local sentence embeddings.

Replaces the SQLite + FTS5 version. The public interface is identical:
  save(user_message, assistant_reply)
  search(query, limit) -> str

Embeddings are generated locally with sentence-transformers (all-MiniLM-L6-v2,
384 dimensions). No extra API key required; the model downloads once (~80 MB).

Requires:
  - Docker running: `docker compose up -d`
  - pip install psycopg2-binary sentence-transformers
"""

import datetime
import threading
from functools import lru_cache

from collectiveos.db import connect, default_user_id

# ---------------------------------------------------------------------------
# Embedding model — loaded in a background thread so startup stays fast.
# If the model isn't ready yet, _embed() raises RuntimeError (callers catch).
# ---------------------------------------------------------------------------

_model_ready = threading.Event()
_model_instance = None
_model_lock = threading.Lock()


def _load_model_bg():
    global _model_instance
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        with _model_lock:
            _model_instance = m
        _model_ready.set()
    except Exception:
        _model_ready.set()  # unblock waiters even on failure


threading.Thread(target=_load_model_bg, daemon=True, name="embed-model-loader").start()


def _embed(text: str) -> list[float]:
    if not _model_ready.wait(timeout=0):
        raise RuntimeError("embedding model not ready yet")
    with _model_lock:
        m = _model_instance
    if m is None:
        raise RuntimeError("embedding model failed to load")
    return m.encode(text, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save(user_message: str, assistant_reply: str, source: str = "conversation") -> None:
    """Embed and persist one exchange to memory, then trigger async entity extraction."""
    try:
        from src import graph_memory as _gm

        content = f"User: {user_message}\nAssistant: {assistant_reply}"
        embedding = _embed(content)
        now = datetime.datetime.utcnow()

        conn = connect()
        try:
            user_id = default_user_id(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_chunks (user_id, source, content, embedding, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (user_id, source, content, embedding, now),
                )
                chunk_id = cur.fetchone()[0]
            conn.commit()
        finally:
            conn.close()

        _gm.trigger_async(chunk_id, content)
    except Exception:
        pass


def save_fact(fact: str) -> None:
    """Save an explicit user fact (preference, name, detail) to memory."""
    try:
        embedding = _embed(fact)
        now = datetime.datetime.utcnow()
        conn = connect()
        try:
            user_id = default_user_id(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memory_chunks (user_id, source, content, embedding, created_at) "
                    "VALUES (%s, 'fact', %s, %s, %s)",
                    (user_id, fact, embedding, now),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def list_facts() -> list[dict]:
    """Return all explicitly saved facts, newest first."""
    conn = connect()
    try:
        user_id = default_user_id(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content, created_at FROM memory_chunks "
                "WHERE user_id = %s AND source = 'fact' ORDER BY created_at DESC",
                (user_id,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {"id": r[0], "content": r[1], "date": r[2].strftime("%Y-%m-%d") if r[2] else ""}
        for r in rows
    ]


def delete_fact(query: str) -> str:
    """
    Delete the fact most semantically similar to *query*.
    Returns the deleted content, or empty string if no facts exist.
    """
    embedding = _embed(query)
    conn = connect()
    try:
        user_id = default_user_id(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content FROM memory_chunks "
                "WHERE user_id = %s AND source = 'fact' "
                "ORDER BY embedding <=> %s::vector LIMIT 1",
                (user_id, embedding),
            )
            row = cur.fetchone()
            if not row:
                return ""
            chunk_id, content = row
            cur.execute("DELETE FROM memory_chunks WHERE id = %s", (chunk_id,))
        conn.commit()
        return content
    finally:
        conn.close()


def get_all_facts_str() -> str:
    """Return all saved facts as a bullet list for injection into the system prompt."""
    facts = list_facts()
    if not facts:
        return ""
    return "\n".join(f"- {f['content']}" for f in facts)


def search(query: str, limit: int = 3) -> str:
    """Return the most semantically similar past exchanges for the query."""
    try:
        embedding = _embed(query)
        conn = connect()
        try:
            user_id = default_user_id(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT content, created_at,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM memory_chunks
                    WHERE user_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, user_id, embedding, limit),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception:
        return ""

    if not rows:
        return ""

    parts = []
    for content, created_at, similarity in rows:
        date = created_at.strftime("%Y-%m-%d") if created_at else ""
        parts.append(f"[{date} | similarity {similarity:.2f}]\n{content}")
    return "\n\n".join(parts)
