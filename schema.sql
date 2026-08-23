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

-- Knowledge graph — entities, mentions, and relationships extracted from conversations
CREATE TABLE IF NOT EXISTS entities (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'concept',  -- person, place, project, organization, concept, product
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, name)
);

CREATE TABLE IF NOT EXISTS entity_mentions (
    id        SERIAL PRIMARY KEY,
    entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    chunk_id  INTEGER REFERENCES memory_chunks(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(entity_id, chunk_id)
);

CREATE TABLE IF NOT EXISTS entity_relations (
    id          SERIAL PRIMARY KEY,
    entity_a_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    entity_b_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    relation    TEXT NOT NULL,
    chunk_id    INTEGER REFERENCES memory_chunks(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Observability — API token usage and tool execution latency
CREATE TABLE IF NOT EXISTS api_usage (
    id            SERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model         TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'chat',
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd      NUMERIC(10, 6) NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id         SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tool_name  TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    success    BOOLEAN NOT NULL DEFAULT TRUE,
    error_type TEXT
);

-- Scheduled routines — prompts that run automatically on a cron schedule
CREATE TABLE IF NOT EXISTS routines (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    schedule    TEXT NOT NULL,   -- cron expression, e.g. '0 8 * * *' = 8am daily
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    notify_via  TEXT NOT NULL DEFAULT 'notification'  -- 'notification' | 'none' | 'telegram'
                CHECK (notify_via IN ('notification', 'none', 'telegram')),
    last_run_at TIMESTAMPTZ,
    last_result TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Health snapshots — metrics pushed from iOS Shortcuts or cached from Oura
CREATE TABLE IF NOT EXISTS health_snapshots (
    id         SERIAL PRIMARY KEY,
    date       DATE NOT NULL,
    source     TEXT NOT NULL DEFAULT 'apple_health',
    metrics    JSONB NOT NULL DEFAULT '{}',   -- steps, sleep_hours, hrv, resting_heart_rate, etc.
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (date, source)
);

-- Scheduled routines: widen notify_via CHECK to include 'telegram' on existing DBs
DO $$
BEGIN
    ALTER TABLE routines DROP CONSTRAINT IF EXISTS routines_notify_via_check;
    ALTER TABLE routines ADD CONSTRAINT routines_notify_via_check
        CHECK (notify_via IN ('notification', 'none', 'telegram'));
EXCEPTION WHEN others THEN NULL;
END $$;

-- Per-connector permission toggles
-- Seeded with all connectors enabled by default.
-- permissions.py creates this table at runtime too (for existing deployments).
CREATE TABLE IF NOT EXISTS connector_permissions (
    connector  TEXT PRIMARY KEY,
    enabled    BOOLEAN NOT NULL DEFAULT TRUE,
    config     JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add config column to existing deployments
ALTER TABLE connector_permissions
    ADD COLUMN IF NOT EXISTS config JSONB NOT NULL DEFAULT '{}';

INSERT INTO connector_permissions (connector) VALUES
    ('memory'),
    ('google_calendar'),
    ('gmail'),
    ('google_drive'),
    ('todoist'),
    ('home_assistant'),
    ('spotify'),
    ('mac_system'),
    ('web_search'),
    ('imessage'),
    ('screen_capture'),
    ('local_files'),
    ('browser'),
    ('contacts'),
    ('reminders'),
    ('notes'),
    ('clipboard'),
    ('telegram'),
    ('notion'),
    ('github'),
    ('slack'),
    ('health'),
    ('finance'),
    ('car'),
    ('appliances')
ON CONFLICT (connector) DO NOTHING;

-- Seed the single default user
INSERT INTO users (name, prefs)
SELECT 'default', '{}'
WHERE NOT EXISTS (SELECT 1 FROM users WHERE name = 'default');

-- -------------------------------------------------------------------------
-- Voice support  (VoiceOS integration — github.com/Anurag9Dhiman/VoiceOS)
-- -------------------------------------------------------------------------

-- Tracks live voice sessions; one row per connected voice-service client.
CREATE TABLE IF NOT EXISTS voice_sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    active_task_ids JSONB NOT NULL DEFAULT '[]'::JSONB,
    entity_stack    JSONB NOT NULL DEFAULT '[]'::JSONB,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS voice_sessions_open_by_user
    ON voice_sessions (user_id)
    WHERE ended_at IS NULL;

-- Tells the voice layer whether a waiting task needs spoken confirmation,
-- a clarification question, or is merely blocked on an external service.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tasks' AND column_name = 'waiting_reason'
    ) THEN
        ALTER TABLE tasks
            ADD COLUMN waiting_reason TEXT
                CHECK (waiting_reason IN ('user_confirm', 'user_clarify', 'external'));
    END IF;
END $$;
