"""
Graph memory — entity extraction and knowledge graph retrieval.

After each conversation save, Haiku extracts named entities and relationships
and stores them in three new tables: entities, entity_mentions, entity_relations.

graph_query(entity_name) returns a human-readable summary of an entity's
neighborhood — its type, all relationships, and conversations where it appeared.

This enriches retrieval beyond pure semantic similarity: a question about
"Alice's project" can pull in chunks mentioning either Alice or the project,
even if those chunks scored below the cosine threshold.
"""

import json
import os
import threading

from google import genai
from google.genai import types as _gtypes

from collectiveos.db import connect, default_user_id

_EXTRACT_MODEL = "models/gemini-flash-latest"
_extract_client: genai.Client | None = None

# ---------------------------------------------------------------------------
# Bootstrap — creates tables if they don't exist (safe for existing DBs)
# ---------------------------------------------------------------------------

_CREATE_ENTITIES = """
CREATE TABLE IF NOT EXISTS entities (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, name)
);
"""

_CREATE_MENTIONS = """
CREATE TABLE IF NOT EXISTS entity_mentions (
    id        SERIAL PRIMARY KEY,
    entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    chunk_id  INTEGER REFERENCES memory_chunks(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(entity_id, chunk_id)
);
"""

_CREATE_RELATIONS = """
CREATE TABLE IF NOT EXISTS entity_relations (
    id          SERIAL PRIMARY KEY,
    entity_a_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    entity_b_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    relation    TEXT NOT NULL,
    chunk_id    INTEGER REFERENCES memory_chunks(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def _bootstrap() -> None:
    try:
        conn = connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE_ENTITIES)
                cur.execute(_CREATE_MENTIONS)
                cur.execute(_CREATE_RELATIONS)
        conn.close()
    except Exception:
        pass


_bootstrap()

# ---------------------------------------------------------------------------
# Haiku extraction
# ---------------------------------------------------------------------------

_SYSTEM = (
    "Extract named entities and relationships from the text. "
    "Return ONLY valid JSON with this exact shape: "
    '{"entities": [{"name": "...", "type": "person|place|project|organization|concept|product"}], '
    '"relations": [{"a": "...", "b": "...", "rel": "brief description"}]}. '
    "Only extract meaningful proper nouns and named concepts — skip generic words. "
    'If nothing meaningful is found, return {"entities": [], "relations": []}.'
)


def _get_client() -> genai.Client:
    global _extract_client
    if _extract_client is None:
        _extract_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _extract_client


def extract_entities(text: str) -> dict:
    """Call Gemini to extract entities and relations from text. Returns parsed dict."""
    try:
        resp = _get_client().models.generate_content(
            model=_EXTRACT_MODEL,
            contents=text[:2000],
            config=_gtypes.GenerateContentConfig(system_instruction=_SYSTEM),
        )
        try:
            from src import observability as _obs
            if resp.usage_metadata:
                _obs.log_api_call(
                    _EXTRACT_MODEL,
                    resp.usage_metadata.prompt_token_count,
                    resp.usage_metadata.candidates_token_count,
                    source="graph_memory",
                )
        except Exception:
            pass
        raw = resp.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except Exception:
        return {"entities": [], "relations": []}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _upsert_entity(cur, user_id: int, name: str, entity_type: str) -> int:
    cur.execute(
        """
        INSERT INTO entities (user_id, name, entity_type)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id, name) DO UPDATE SET entity_type = EXCLUDED.entity_type
        RETURNING id
        """,
        (user_id, name.lower().strip(), entity_type),
    )
    return cur.fetchone()[0]


def save_entities(chunk_id: int, extracted: dict) -> None:
    """Persist extracted entities and relations for a memory chunk."""
    entities = extracted.get("entities", [])
    relations = extracted.get("relations", [])
    if not entities:
        return

    conn = connect()
    try:
        user_id = default_user_id(conn)
        name_to_id: dict[str, int] = {}
        with conn.cursor() as cur:
            for e in entities:
                name = (e.get("name") or "").strip()
                etype = e.get("type", "concept")
                if not name:
                    continue
                eid = _upsert_entity(cur, user_id, name, etype)
                name_to_id[name.lower()] = eid
                cur.execute(
                    "INSERT INTO entity_mentions (entity_id, chunk_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (eid, chunk_id),
                )
            for r in relations:
                a = (r.get("a") or "").lower().strip()
                b = (r.get("b") or "").lower().strip()
                rel = (r.get("rel") or "").strip()[:120]
                if a not in name_to_id or b not in name_to_id or not rel:
                    continue
                cur.execute(
                    "INSERT INTO entity_relations (entity_a_id, entity_b_id, relation, chunk_id) VALUES (%s, %s, %s, %s)",
                    (name_to_id[a], name_to_id[b], rel, chunk_id),
                )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def extract_and_save(chunk_id: int, text: str) -> None:
    """Extract entities from text and persist — designed to run in a background thread."""
    extracted = extract_entities(text)
    save_entities(chunk_id, extracted)


def trigger_async(chunk_id: int, text: str) -> None:
    """Fire-and-forget entity extraction — does not block the agent loop."""
    threading.Thread(
        target=extract_and_save, args=(chunk_id, text), daemon=True
    ).start()


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_graph(entity_name: str) -> str:
    """
    Return a human-readable summary of an entity's knowledge graph neighborhood:
    its type, all known relationships, and the conversations where it appeared.
    """
    conn = connect()
    try:
        user_id = default_user_id(conn)
        with conn.cursor() as cur:
            # Find entity by fuzzy name match
            cur.execute(
                "SELECT id, name, entity_type FROM entities "
                "WHERE user_id = %s AND name ILIKE %s "
                "ORDER BY created_at DESC LIMIT 1",
                (user_id, f"%{entity_name.strip().lower()}%"),
            )
            row = cur.fetchone()
            if not row:
                return f"No entity found matching '{entity_name}'."
            eid, name, etype = row

            # Relations where this entity appears on either side
            cur.execute(
                """
                SELECT e_other.name, er.relation,
                       (er.entity_a_id = %s) AS is_source
                FROM entity_relations er
                JOIN entities e_other ON e_other.id = CASE
                    WHEN er.entity_a_id = %s THEN er.entity_b_id
                    ELSE er.entity_a_id
                END
                WHERE er.entity_a_id = %s OR er.entity_b_id = %s
                ORDER BY er.created_at DESC
                LIMIT 20
                """,
                (eid, eid, eid, eid),
            )
            rels = cur.fetchall()

            # Chunks where this entity is mentioned
            cur.execute(
                """
                SELECT mc.content, mc.created_at
                FROM entity_mentions em
                JOIN memory_chunks mc ON mc.id = em.chunk_id
                WHERE em.entity_id = %s
                ORDER BY mc.created_at DESC
                LIMIT 5
                """,
                (eid,),
            )
            chunks = cur.fetchall()
    finally:
        conn.close()

    lines = [f"Entity: {name.title()} ({etype})"]

    if rels:
        lines.append("\nRelationships:")
        for other_name, relation, is_source in rels:
            if is_source:
                lines.append(f"  {name.title()} → {relation} → {other_name.title()}")
            else:
                lines.append(f"  {other_name.title()} → {relation} → {name.title()}")

    if chunks:
        lines.append(f"\nMentioned in {len(chunks)} conversation(s):")
        for content, created_at in chunks:
            date = created_at.strftime("%Y-%m-%d") if created_at else ""
            preview = content[:150].replace("\n", " ")
            lines.append(f"  [{date}] {preview}…")

    if not rels and not chunks:
        lines.append("  (no relationships or conversation mentions recorded yet)")

    return "\n".join(lines)
