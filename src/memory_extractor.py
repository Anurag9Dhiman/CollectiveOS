"""
memory_extractor.py
-------------------
Automatic fact extraction from conversation exchanges.

After every conversation turn, this module runs in a background daemon
thread, asks Gemini to identify durable facts worth remembering, deduplicates
against what is already stored, and writes new facts with source='auto_fact'.

No external packages beyond google-genai are required — no LangMem dependency.

Public API:
    trigger(user_message, assistant_reply)   — fire-and-forget (non-blocking)
    extract(user_message, assistant_reply)   → list[str]  (synchronous, for tests)
"""

from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger(__name__)

_EXTRACT_PROMPT = """You are a personal assistant memory system.
Read the conversation exchange below and extract any durable facts about the USER
that are worth remembering for future conversations.

Only extract facts that are:
  - Personal preferences (food, music, tools, workflows, communication style)
  - Important people in their life (names, roles, relationships)
  - Ongoing projects, goals, or responsibilities
  - Skills, expertise, or background they mentioned
  - Explicit decisions they made or plans they stated
  - Things they explicitly want remembered

Do NOT extract:
  - One-off questions with no lasting relevance
  - General knowledge or chitchat
  - Temporary states ("I'm tired today")
  - Facts already obvious from context

Reply ONLY with a JSON array of strings — one fact per string, max 10 facts.
If nothing is worth remembering, reply with an empty array: []

Example: ["User prefers dark mode in all apps.", "User is building a project called CollectiveOS."]
"""


def extract(user_message: str, assistant_reply: str) -> list[str]:
    """Synchronous extraction — returns list of fact strings."""
    import json
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return []

    model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    exchange = (
        f"User: {user_message[:800]}\n"
        f"Assistant: {assistant_reply[:800]}"
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[types.Part(text=exchange)],
            config=types.GenerateContentConfig(
                system_instruction=_EXTRACT_PROMPT,
                temperature=0.1,
                max_output_tokens=512,
            ),
        )
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        facts = json.loads(text)
        return [f for f in facts if isinstance(f, str) and f.strip()]
    except Exception as exc:
        log.debug("memory_extractor.extract failed: %s", exc)
        return []


def _save_auto_facts(facts: list[str]) -> None:
    """Persist extracted facts, skipping near-duplicates of existing ones."""
    if not facts:
        return

    try:
        from src.memory import connect, default_user_id, _embed, list_facts
        import datetime

        existing_texts = {f["content"].lower() for f in list_facts()}
        # Also load auto_facts for dedup
        conn = connect()
        try:
            user_id = default_user_id(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content FROM memory_chunks "
                    "WHERE user_id = %s AND source = 'auto_fact'",
                    (user_id,),
                )
                for row in cur.fetchall():
                    existing_texts.add(row[0].lower())

            now = datetime.datetime.utcnow()
            saved = 0
            for fact in facts:
                # Skip if very similar text already exists (simple substring check)
                key = fact.lower()
                if any(key in ex or ex in key for ex in existing_texts):
                    continue
                embedding = _embed(fact)
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO memory_chunks "
                        "(user_id, source, content, embedding, created_at) "
                        "VALUES (%s, 'auto_fact', %s, %s::vector, %s)",
                        (user_id, fact, embedding, now),
                    )
                existing_texts.add(key)
                saved += 1
            conn.commit()
            if saved:
                log.info("memory_extractor: saved %d auto facts", saved)
        finally:
            conn.close()
    except Exception as exc:
        log.debug("memory_extractor._save_auto_facts failed: %s", exc)


def trigger(user_message: str, assistant_reply: str) -> None:
    """Non-blocking: extract and save facts in a background daemon thread."""
    def _run():
        facts = extract(user_message, assistant_reply)
        _save_auto_facts(facts)

    threading.Thread(target=_run, daemon=True, name="mem-extract").start()
