"""
Conversation auto-titler — generates a short title for a conversation
using a cheap Gemini Flash call, then persists it to the DB.

Called in a background thread after POST /chat completes so it never
blocks the user's response. Only titles once per conversation (skips if
a title is already set).
"""

import logging
import os
import threading

log = logging.getLogger(__name__)

_TITLE_MODEL = os.environ.get("GEMINI_ROUTER_MODEL", "gemini-2.0-flash-lite")
_TITLE_PROMPT = (
    "Generate a short, specific title (4–7 words, no quotes, no punctuation at the end) "
    "that captures what this conversation is about. "
    "Respond with only the title — nothing else.\n\n"
    "Conversation:\n{snippet}"
)


def _call_gemini(snippet: str) -> str:
    import google.genai as genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return ""
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=_TITLE_MODEL,
        contents=_TITLE_PROMPT.format(snippet=snippet),
    )
    return (resp.text or "").strip().strip('"').strip("'")


def generate_title(messages: list[dict]) -> str:
    """
    Given a list of {role, content} dicts, ask Gemini for a short title.
    Returns an empty string on any error.
    """
    # Build a compact snippet: first user message + first assistant reply
    parts = []
    for m in messages[:6]:
        if m.get("role") in ("user", "assistant"):
            label = "User" if m["role"] == "user" else "Assistant"
            content = (m.get("content") or "")[:300]
            parts.append(f"{label}: {content}")
        if len(parts) >= 4:
            break
    if not parts:
        return ""
    snippet = "\n".join(parts)
    try:
        title = _call_gemini(snippet)
        # Sanity-check: reject titles that are too long or empty
        if not title or len(title) > 100:
            return ""
        return title
    except Exception as exc:
        log.debug("Title generation failed: %s", exc)
        return ""


def title_conversation(conversation_id: int) -> None:
    """
    Generate and store a title for a conversation — skips if already titled.
    Safe to call multiple times; idempotent.
    """
    from src.conversations import load_history, set_title, get_title
    try:
        existing = get_title(conversation_id)
        if existing:
            return  # already titled
        messages = load_history(conversation_id, limit=6)
        if not messages:
            return
        title = generate_title(messages)
        if title:
            set_title(conversation_id, title)
            log.debug("Titled conversation %d: %r", conversation_id, title)
    except Exception as exc:
        log.debug("title_conversation(%d) error: %s", conversation_id, exc)


def title_async(conversation_id: int) -> None:
    """Fire-and-forget title generation in a daemon thread."""
    threading.Thread(
        target=title_conversation,
        args=(conversation_id,),
        daemon=True,
    ).start()
