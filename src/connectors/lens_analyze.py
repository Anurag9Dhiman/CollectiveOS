"""VisualOS connector — analyze images via the Lens OS REST API.

Risk tier: L0 (read-only, no side effects).
Configure in .env:
  LENS_URL=http://localhost:7000       (default — where VisualOS runs)
  LENS_API_KEY=<your-lens-api-key>
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

_LENS_URL = os.environ.get("LENS_URL", "http://localhost:7000")
_LENS_API_KEY = os.environ.get("LENS_API_KEY", "")
_TIMEOUT = 15.0

_MIME_MAP = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


def lens_analyze(
    image_path: str,
    lat: float | None = None,
    lng: float | None = None,
    user_id: str = "default",
) -> str:
    """Identify and describe what's in an image using VisualOS.

    Returns a text summary with headline, body, and source citations.
    image_path: absolute or CWD-relative path to a JPEG, PNG, or WebP file.
    lat/lng: optional GPS coordinates to enrich the result with location context.
    """
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        return f"Error: image not found at {image_path}"

    mime = _MIME_MAP.get(path.suffix.lstrip(".").lower(), "image/jpeg")
    headers = {"X-API-Key": _LENS_API_KEY} if _LENS_API_KEY else {}

    form: dict[str, str] = {"user_id": user_id}
    if lat is not None:
        form["lat"] = str(lat)
    if lng is not None:
        form["lng"] = str(lng)

    with open(path, "rb") as fh:
        image_bytes = fh.read()

    try:
        resp = httpx.post(
            f"{_LENS_URL}/analyze",
            files={"image": (path.name, image_bytes, mime)},
            data=form,
            headers=headers,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return f"VisualOS returned {exc.response.status_code}: {exc.response.text[:300]}"
    except httpx.ConnectError:
        return "VisualOS is not reachable. Is it running? (LENS_URL defaults to http://localhost:7000)"
    except Exception as exc:
        return f"Error calling VisualOS: {exc}"

    return _format_card(resp.json())


def _format_card(card: dict) -> str:
    if card.get("card_type") == "fallback":
        parts = [card.get("headline", "Unknown")]
        if obs := card.get("observation"):
            parts.append(obs)
        if sug := card.get("suggestion"):
            parts.append(f"Suggestion: {sug}")
        return " — ".join(parts)

    headline = card.get("headline", "")
    body = card.get("body", "")
    citations = [c["source_name"] for c in card.get("citations", []) if c.get("source_name")]
    suffix = f"\n\nSources: {', '.join(citations)}" if citations else ""
    return f"{headline}\n\n{body}{suffix}"
