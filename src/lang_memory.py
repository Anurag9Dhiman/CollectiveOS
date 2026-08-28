"""LangMem-backed memory extraction for CollectiveOS.

Uses LangMem's create_memory_manager to extract durable, structured facts
from conversation exchanges before they are stored. This replaces the
previous approach of embedding raw conversation text: instead of storing
"User: I like coffee\nAssistant: Got it", we store "User prefers coffee,
specifically dark roast."

Public interface:
    extract_facts(user_message, assistant_reply, existing) → list[(id, text)]

The returned list follows LangMem's insert/update protocol:
  - id in existing_ids → update that existing fact (LangMem deduplicates)
  - id not in existing_ids → insert as a new fact

Falls back gracefully (returns empty list) if the LLM call fails, so the
caller's raw-text fallback in memory.save() still runs.

The manager is a process-lifetime singleton; the underlying LangChain
ChatGoogleGenerativeAI client is thread-safe.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("collectiveos.lang_memory")

_manager = None


def _get_manager():
    global _manager
    if _manager is not None:
        return _manager

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langmem import create_memory_manager

    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
        google_api_key=os.environ.get("GEMINI_API_KEY", ""),
        temperature=0,
    )

    _manager = create_memory_manager(
        llm,
        instructions=(
            "Extract durable, specific facts about the user from this conversation. "
            "Focus only on: personal preferences, habits, important relationships, "
            "ongoing projects, goals, and facts the user explicitly shared. "
            "Skip one-off questions and chitchat with no lasting value. "
            "Write each fact as a concise, standalone sentence."
        ),
        enable_inserts=True,
        enable_updates=True,
        enable_deletes=False,
    )
    return _manager


def extract_facts(
    user_message: str,
    assistant_reply: str,
    existing: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Extract key facts from one conversation exchange.

    Args:
        user_message: The user's turn.
        assistant_reply: The assistant's response.
        existing: Current facts as [(str_id, content), ...].
                  str_id should be the Postgres row ID as a string.

    Returns:
        List of (id, content) pairs.
        id matches an existing str_id → this is an UPDATE.
        id is a new UUID → this is an INSERT.
        Empty list on failure (caller falls back to raw storage).
    """
    try:
        from langchain_core.messages import AIMessage, HumanMessage
        from langmem.knowledge.extraction import Memory

        manager = _get_manager()

        existing_memories = [(id_, Memory(content=text)) for id_, text in existing]

        result = manager.invoke({
            "messages": [
                HumanMessage(content=user_message),
                AIMessage(content=assistant_reply),
            ],
            "existing": existing_memories,
        })

        # ExtractedMemory is NamedTuple(id: str, content: Memory)
        extracted = []
        for item in result:
            content_obj = item.content
            # content is a Memory instance (or similar schema); get the string
            text = (
                content_obj.content
                if hasattr(content_obj, "content")
                else str(content_obj)
            )
            extracted.append((item.id, text))

        logger.debug("LangMem extracted %d facts from exchange", len(extracted))
        return extracted

    except Exception as exc:
        logger.warning("LangMem extraction failed (%s) — raw fallback will run", exc)
        return []
