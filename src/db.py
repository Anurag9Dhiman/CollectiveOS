"""
Postgres connection management.

Reads DATABASE_URL from the environment. Falls back to a default local URL
that matches the docker-compose.yml credentials so `docker compose up` works
with zero config.
"""

import os
import psycopg2
import psycopg2.extras

DEFAULT_URL = "postgresql://assistant:assistant@localhost:5432/assistant"


def connect() -> psycopg2.extensions.connection:
    url = os.environ.get("DATABASE_URL", DEFAULT_URL)
    conn = psycopg2.connect(url)
    conn.autocommit = False
    return conn


def default_user_id(conn) -> int:
    """Return the id of the single default user row seeded by schema.sql."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE name = 'default' LIMIT 1")
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute("INSERT INTO users (name, prefs) VALUES ('default', '{}') RETURNING id")
        conn.commit()
        return cur.fetchone()[0]
