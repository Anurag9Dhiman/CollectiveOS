"""
Data exporter — assembles all user data into a single serialisable dict.

Used by GET /export to produce a JSON backup the user can download.
Each section can be included or excluded via the `include` set.

Sections:
  conversations  — last 500 conversations with all their messages
  facts          — explicit memory facts
  entities       — knowledge graph entities and relations
  routines       — scheduled routines
  watchers       — proactive condition watchers
"""

from __future__ import annotations

import datetime
from typing import Any

ALL_SECTIONS = {"conversations", "facts", "entities", "routines", "watchers"}


def build(include: set[str] | None = None) -> dict[str, Any]:
    """
    Assemble export data.  *include* is a set of section names; defaults to all.
    Returns a dict safe to pass to json.dumps / JSONResponse.
    """
    if include is None:
        include = ALL_SECTIONS

    result: dict[str, Any] = {
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "version": 1,
    }

    if "conversations" in include:
        result["conversations"] = _export_conversations()

    if "facts" in include:
        result["facts"] = _export_facts()

    if "entities" in include:
        result["entities"] = _export_entities()

    if "routines" in include:
        result["routines"] = _export_routines()

    if "watchers" in include:
        result["watchers"] = _export_watchers()

    return result


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _export_conversations() -> list[dict]:
    """Return the 500 most recent conversations with all their messages."""
    from src.db import connect
    conn = connect()
    try:
        with conn.cursor() as cur:
            # Fetch conversations
            cur.execute(
                """SELECT id, title, started_at
                   FROM conversations
                   ORDER BY started_at DESC
                   LIMIT 500""",
            )
            convs = {r[0]: {"id": r[0], "title": r[1],
                             "started_at": r[2].isoformat() if r[2] else None,
                             "messages": []}
                     for r in cur.fetchall()}

            if not convs:
                return []

            # Bulk-fetch all messages for those conversations in one query
            ids = tuple(convs.keys())
            cur.execute(
                """SELECT conversation_id, role, content, created_at
                   FROM messages
                   WHERE conversation_id = ANY(%s)
                   ORDER BY created_at ASC""",
                (list(ids),),
            )
            for row in cur.fetchall():
                conv_id, role, content, created_at = row
                if conv_id in convs:
                    convs[conv_id]["messages"].append({
                        "role": role,
                        "content": content,
                        "created_at": created_at.isoformat() if created_at else None,
                    })
        return list(convs.values())
    finally:
        conn.close()


def _export_facts() -> list[dict]:
    from src import memory
    return memory.list_facts()


def _export_entities() -> dict:
    from src.db import connect
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT e.id, e.name, e.entity_type,
                          COUNT(em.chunk_id) AS mention_count
                   FROM entities e
                   LEFT JOIN entity_mentions em ON em.entity_id = e.id
                   GROUP BY e.id, e.name, e.entity_type
                   ORDER BY mention_count DESC, e.name"""
            )
            nodes = [
                {"id": r[0], "name": r[1], "type": r[2], "mention_count": r[3]}
                for r in cur.fetchall()
            ]
            cur.execute(
                """SELECT ea.name, er.relation, eb.name
                   FROM entity_relations er
                   JOIN entities ea ON ea.id = er.entity_a_id
                   JOIN entities eb ON eb.id = er.entity_b_id"""
            )
            edges = [
                {"from": r[0], "relation": r[1], "to": r[2]}
                for r in cur.fetchall()
            ]
        return {"nodes": nodes, "edges": edges}
    finally:
        conn.close()


def _export_routines() -> list[dict]:
    from src import routines
    return routines.list_all()


def _export_watchers() -> list[dict]:
    from src import watchers
    return watchers.list_all()
