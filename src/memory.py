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
from functools import lru_cache

from src.db import connect, default_user_id

# ---------------------------------------------------------------------------
# Embedding model (loaded once, reused across calls)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _embed(text: str) -> list[float]:
    return _model().encode(text, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save(user_message: str, assistant_reply: str, source: str = "conversation") -> None:
    """Embed and persist one exchange to memory."""
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
                """,
                (user_id, source, content, embedding, now),
            )
        conn.commit()
    finally:
        conn.close()


def save_fact(fact: str) -> None:
    """Save an explicit user fact (preference, name, detail) to memory."""
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

    if not rows:
        return ""

    parts = []
    for content, created_at, similarity in rows:
        date = created_at.strftime("%Y-%m-%d") if created_at else ""
        parts.append(f"[{date} | similarity {similarity:.2f}]\n{content}")
    return "\n\n".join(parts)
