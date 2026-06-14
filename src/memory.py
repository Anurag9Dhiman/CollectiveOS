"""
SQLite-backed memory store.

Saves every conversation exchange and retrieves relevant past context using
SQLite's built-in FTS5 full-text search. Drop-in replacement with Postgres +
pgvector later — only this file changes.

Schema
------
  memory_chunks(id, source, content, created_at)

  FTS virtual table (memory_fts) mirrors content for keyword search.
"""

import os
import sqlite3
import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
DB_PATH = os.path.join(_ROOT, "memory.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memory_chunks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            source     TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            created_at TEXT    NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
        USING fts5(content, content='memory_chunks', content_rowid='id');

        CREATE TRIGGER IF NOT EXISTS memory_chunks_ai
        AFTER INSERT ON memory_chunks BEGIN
            INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
        END;
    """)
    conn.commit()


def save(user_message: str, assistant_reply: str, source: str = "conversation") -> None:
    """Persist one exchange to memory."""
    content = f"User: {user_message}\nAssistant: {assistant_reply}"
    now = datetime.datetime.utcnow().isoformat()
    with _connect() as conn:
        _init_db(conn)
        conn.execute(
            "INSERT INTO memory_chunks (source, content, created_at) VALUES (?, ?, ?)",
            (source, content, now),
        )
        conn.commit()


def search(query: str, limit: int = 3) -> str:
    """Return the most relevant past exchanges for the given query string."""
    with _connect() as conn:
        _init_db(conn)
        rows = conn.execute(
            """
            SELECT mc.content, mc.created_at
            FROM memory_fts
            JOIN memory_chunks mc ON memory_fts.rowid = mc.id
            WHERE memory_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()

    if not rows:
        return ""

    parts = []
    for row in rows:
        parts.append(f"[{row['created_at'][:10]}]\n{row['content']}")
    return "\n\n".join(parts)
