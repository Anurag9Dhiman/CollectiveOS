"""
Memory store — Postgres + pgvector with Gemini embeddings.

Embeddings are generated via the Gemini Embedding API (gemini-embedding-001,
3072 dimensions). This replaces the local sentence-transformers model, which
caused a PyTorch mutex deadlock under async load and produced 384-dim vectors
mismatched with VisualOS's embedding model.

Public interface is identical to the previous version — no call-site changes.
"""

import datetime
import os

from src.db import connect, default_user_id

_EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
_EMBED_DIM = 3072  # gemini-embedding-001 output dimension


def _embed(text: str) -> list[float]:
    """Embed text using the Gemini Embedding API. Synchronous — safe to call from threads."""
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.embed_content(
        model=_EMBED_MODEL,
        contents=text,
    )
    return list(resp.embeddings[0].values)


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
                    VALUES (%s, %s, %s, %s::vector, %s)
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


def save_smart(user_message: str, assistant_reply: str) -> None:
    """Extract structured facts via LangMem then persist them, with raw fallback.

    Preferred over save() for new conversations: LangMem identifies what is
    actually worth remembering (preferences, facts, relationships) and
    deduplicates against existing facts before writing.

    Falls back to save() if LangMem extraction returns nothing or fails.
    """
    try:
        from src import lang_memory as _lm

        existing = [(str(f["id"]), f["content"]) for f in list_facts()]
        existing_ids = {e[0] for e in existing}

        extracted = _lm.extract_facts(user_message, assistant_reply, existing)

        if not extracted:
            save(user_message, assistant_reply)
            return

        now = datetime.datetime.utcnow()
        conn = connect()
        try:
            user_id = default_user_id(conn)
            with conn.cursor() as cur:
                for fact_id, content in extracted:
                    embedding = _embed(content)
                    if fact_id in existing_ids:
                        cur.execute(
                            "UPDATE memory_chunks "
                            "SET content = %s, embedding = %s::vector, created_at = %s "
                            "WHERE id::text = %s AND user_id = %s AND source = 'fact'",
                            (content, embedding, now, fact_id, user_id),
                        )
                    else:
                        cur.execute(
                            "INSERT INTO memory_chunks "
                            "(user_id, source, content, embedding, created_at) "
                            "VALUES (%s, 'fact', %s, %s::vector, %s)",
                            (user_id, content, embedding, now),
                        )
            conn.commit()
        finally:
            conn.close()

        # Still store the raw exchange for semantic search context
        save(user_message, assistant_reply)

    except Exception:
        save(user_message, assistant_reply)


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
                    "VALUES (%s, 'fact', %s, %s::vector, %s)",
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
    """Delete the fact most semantically similar to *query*.

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
