"""
Conversation persistence — stores and retrieves message history from Postgres.

Tables used: conversations, messages (see schema.sql).
Public API:
  create()                      -> int  (conversation id)
  save_message(conv_id, role, content)
  load_history(conv_id, limit)  -> list[dict]
"""

from src.db import connect, default_user_id


def create() -> int:
    """Insert a new conversation row and return its id."""
    conn = connect()
    try:
        user_id = default_user_id(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (user_id) VALUES (%s) RETURNING id",
                (user_id,),
            )
            conv_id = cur.fetchone()[0]
        conn.commit()
        return conv_id
    finally:
        conn.close()


def save_message(conversation_id: int, role: str, content: str) -> None:
    """Append a single message to the conversation."""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
                (conversation_id, role, content),
            )
        conn.commit()
    finally:
        conn.close()


def get_title(conversation_id: int) -> str | None:
    """Return the stored title for a conversation, or None if not yet set."""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT title FROM conversations WHERE id = %s", (conversation_id,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()


def set_title(conversation_id: int, title: str) -> None:
    """Persist a generated title on the conversation row."""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET title = %s WHERE id = %s AND title IS NULL",
                (title[:200], conversation_id),
            )
        conn.commit()
    finally:
        conn.close()


def list_conversations(limit: int = 50) -> list[dict]:
    """
    Return the most recent *limit* conversations, newest first.
    Each dict: {id, title, started_at, first_message (str|None), message_count}.
    """
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id,
                       c.started_at,
                       c.title,
                       (SELECT content FROM messages
                        WHERE conversation_id = c.id AND role = 'user'
                        ORDER BY created_at ASC LIMIT 1) AS first_message,
                       (SELECT COUNT(*) FROM messages
                        WHERE conversation_id = c.id) AS message_count
                FROM conversations c
                ORDER BY c.started_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
            return [
                {
                    "id": r[0],
                    "started_at": r[1].isoformat() if r[1] else None,
                    "title": r[2],
                    "first_message": r[3],
                    "message_count": r[4],
                }
                for r in rows
            ]
    finally:
        conn.close()


def search_messages(query: str, limit: int = 20) -> list[dict]:
    """
    Full-text search across all message content using Postgres tsvector.

    Returns up to *limit* results, ranked by relevance, each with:
      {conversation_id, started_at, role, snippet, rank}

    Falls back to ILIKE if the query contains characters that break
    plainto_tsquery (e.g. very short words or pure punctuation).
    """
    if not query or not query.strip():
        return []
    conn = connect()
    try:
        with conn.cursor() as cur:
            # Try tsvector first; fall back to ILIKE on error
            try:
                cur.execute(
                    """
                    SELECT m.conversation_id,
                           c.started_at,
                           m.role,
                           ts_headline(
                               'english', m.content,
                               plainto_tsquery('english', %s),
                               'MaxWords=20, MinWords=10, ShortWord=2,
                                StartSel=<mark>, StopSel=</mark>'
                           ) AS snippet,
                           ts_rank(
                               to_tsvector('english', m.content),
                               plainto_tsquery('english', %s)
                           ) AS rank
                    FROM messages m
                    JOIN conversations c ON c.id = m.conversation_id
                    WHERE m.role IN ('user', 'assistant')
                      AND to_tsvector('english', m.content)
                          @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (query, query, query, limit),
                )
            except Exception:
                conn.rollback()
                # Fallback: plain ILIKE with a simple excerpt
                like = f"%{query}%"
                cur.execute(
                    """
                    SELECT m.conversation_id,
                           c.started_at,
                           m.role,
                           SUBSTRING(m.content FOR 120) AS snippet,
                           1.0 AS rank
                    FROM messages m
                    JOIN conversations c ON c.id = m.conversation_id
                    WHERE m.role IN ('user', 'assistant')
                      AND m.content ILIKE %s
                    ORDER BY c.started_at DESC
                    LIMIT %s
                    """,
                    (like, limit),
                )
            rows = cur.fetchall()
            return [
                {
                    "conversation_id": r[0],
                    "started_at": r[1].isoformat() if r[1] else None,
                    "role": r[2],
                    "snippet": r[3],
                    "rank": float(r[4]),
                }
                for r in rows
            ]
    finally:
        conn.close()


def load_history(conversation_id: int, limit: int = 20) -> list[dict]:
    """
    Return the last *limit* messages from a conversation, oldest first.
    Each dict has keys: role, content.
    Returns [] if the conversation doesn't exist or has no messages.
    """
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM (
                    SELECT role, content, created_at
                    FROM messages
                    WHERE conversation_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) sub
                ORDER BY created_at ASC
                """,
                (conversation_id, limit),
            )
            return [{"role": row[0], "content": row[1]} for row in cur.fetchall()]
    finally:
        conn.close()
