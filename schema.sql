-- CollectiveOS — Postgres schema
-- Run automatically by Docker on first start, or manually:
--   psql $DATABASE_URL -f schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

-- Users (single-user for now; schema supports multi-user later)
CREATE TABLE IF NOT EXISTS users (
    id         SERIAL PRIMARY KEY,
    name       TEXT NOT NULL,
    prefs      JSONB NOT NULL DEFAULT '{}'
);

-- Conversations
CREATE TABLE IF NOT EXISTS conversations (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Messages within a conversation
CREATE TABLE IF NOT EXISTS messages (
    id              SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tasks the assistant is executing
CREATE TABLE IF NOT EXISTS tasks (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','planning','running','waiting','blocked','completed','failed','cancelled')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Individual steps within a task
CREATE TABLE IF NOT EXISTS task_steps (
    id        SERIAL PRIMARY KEY,
    task_id   INTEGER REFERENCES tasks(id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL,
    input     JSONB NOT NULL DEFAULT '{}',
    output    TEXT,
    status    TEXT NOT NULL DEFAULT 'pending'
              CHECK (status IN ('pending','running','completed','failed'))
);

-- External connectors (Calendar, Gmail, Drive, ...)
CREATE TABLE IF NOT EXISTS connectors (
    id      SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name    TEXT NOT NULL,
    type    TEXT NOT NULL,
    status  TEXT NOT NULL DEFAULT 'active'
);

-- OAuth tokens and secrets per connector
CREATE TABLE IF NOT EXISTS credentials (
    id           SERIAL PRIMARY KEY,
    connector_id INTEGER REFERENCES connectors(id) ON DELETE CASCADE,
    token_ref    TEXT NOT NULL,
    expires_at   TIMESTAMPTZ
);

-- Devices reachable through connectors
CREATE TABLE IF NOT EXISTS devices (
    id           SERIAL PRIMARY KEY,
    connector_id INTEGER REFERENCES connectors(id) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    type         TEXT NOT NULL,
    state        JSONB NOT NULL DEFAULT '{}'
);

-- Memory chunks — text + embedding for semantic search
CREATE TABLE IF NOT EXISTS memory_chunks (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
    source     TEXT NOT NULL DEFAULT 'conversation',
    content    TEXT NOT NULL,
    embedding  vector(384),           -- matches all-MiniLM-L6-v2 output dim
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS memory_chunks_embedding_idx
    ON memory_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Per-connector permission toggles
-- Seeded with all connectors enabled by default.
-- permissions.py creates this table at runtime too (for existing deployments).
CREATE TABLE IF NOT EXISTS connector_permissions (
    connector  TEXT PRIMARY KEY,
    enabled    BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO connector_permissions (connector) VALUES
    ('google_calendar'),
    ('gmail'),
    ('google_drive'),
    ('todoist'),
    ('home_assistant'),
    ('spotify'),
    ('mac_system'),
    ('web_search')
ON CONFLICT (connector) DO NOTHING;

-- Seed the single default user
INSERT INTO users (name, prefs)
SELECT 'default', '{}'
WHERE NOT EXISTS (SELECT 1 FROM users WHERE name = 'default');
