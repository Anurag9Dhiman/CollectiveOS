"""VisualOS agent client.

Wraps the VisualOS (Lens OS) REST + A2A API.

Primary use-cases inside CollectiveOS:
  1. Context enrichment — when VoiceOS passes a scan_session_id in entity_refs,
     fetch the stored ScanContext so the task agent can answer follow-ups.
  2. Direct image analysis — POST /analyze with a screenshot (used by
     capture_screen connector; not called from here).
  3. A2A queries — future: send tasks via the JSON-RPC 2.0 endpoint.

The client never raises; errors are returned as AgentResult with an
error-prefix text so the orchestrator can handle them gracefully.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error

from .base import AgentClient, AgentResult

logger = logging.getLogger("collectiveos.agents.visual")


class VisualOSClient(AgentClient):
    def __init__(self, base_url: str, api_key: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    @property
    def capabilities(self) -> list[str]:
        return ["visual", "screen_analysis", "image_qa", "scan_context"]

    def call(self, text: str, context: dict | None = None, **_) -> AgentResult:
        """Enrich with an existing scan session, or signal no visual context."""
        session_id = (context or {}).get("scan_session_id")
        if session_id:
            return self._fetch_scan_context(session_id)
        return AgentResult(text="")   # no visual context — orchestrator handles this

    def _fetch_scan_context(self, session_id: str) -> AgentResult:
        """GET /session/{id} and format it as a concise context string."""
        url = f"{self._base_url}/session/{session_id}"
        headers: dict[str, str] = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            entity = data.get("entity_name", "")
            headline = data.get("card_headline", "")
            body = data.get("card_body", "")
            hist_facts = "; ".join(
                f["fact"] for f in data.get("historical_facts", [])[:3]
            )
            live_facts = "; ".join(
                f["fact"] for f in data.get("live_facts", [])[:2]
            )
            nearby = data.get("nearby_context", "")

            parts = [f"Entity: {entity}", f"Headline: {headline}"]
            if body:
                parts.append(f"Description: {body}")
            if hist_facts:
                parts.append(f"Historical facts: {hist_facts}")
            if live_facts:
                parts.append(f"Live facts: {live_facts}")
            if nearby:
                parts.append(f"Nearby context: {nearby}")

            return AgentResult(
                text="\n".join(parts),
                session_id=session_id,
                metadata=data,
            )

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return AgentResult(text="")   # session expired — no context
            logger.warning("VisualOS session fetch error %s: %s", session_id, exc)
            return AgentResult(text="")
        except Exception as exc:
            logger.warning("VisualOS unreachable: %s", exc)
            return AgentResult(text="")

    def analyze_image(self, image_bytes: bytes, question: str = "") -> AgentResult:
        """POST /analyze — direct image analysis (stdlib multipart, no deps)."""
        boundary = "----CollectiveOSBoundary"
        parts: list[bytes] = []

        def _field(name: str, value: str) -> bytes:
            return (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()

        parts.append(_field("user_id", "collectiveos"))
        if question.strip():
            parts.append(_field("query", question.strip()))
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="screen.png"\r\n'
            f"Content-Type: image/png\r\n\r\n".encode()
            + image_bytes
            + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)

        headers: dict[str, str] = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        }
        if self._api_key:
            headers["X-API-Key"] = self._api_key

        try:
            req = urllib.request.Request(
                f"{self._base_url}/analyze",
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            card = data.get("card", {})
            session_id = data.get("session_id")
            card_type = card.get("card_type", "fallback")
            if card_type == "normal":
                text = f"{card.get('headline', '')}\n\n{card.get('body', '')}".strip()
            else:
                text = "\n\n".join(
                    p for p in [card.get("headline"), card.get("observation"), card.get("suggestion")] if p
                ) or "VisualOS returned an empty response."

            return AgentResult(text=text, session_id=session_id, metadata=card)

        except Exception as exc:
            logger.warning("VisualOS /analyze error: %s", exc)
            return AgentResult(text=f"[VisualOS unavailable: {exc}]")

    def is_healthy(self) -> bool:
        try:
            req = urllib.request.Request(f"{self._base_url}/health")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False
