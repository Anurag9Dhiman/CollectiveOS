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
