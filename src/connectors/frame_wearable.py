"""
Frame wearable connector — Brilliant Labs Frame glasses.

Two communication paths:

  WebSocket path (primary, no extra deps):
    - Frame companion app sends camera frames to CollectiveOS via /wearable/ws
    - CollectiveOS processes the task and sends formatted reply back over the socket
    - Companion app receives the reply and shows it on the in-lens display via BLE

  Direct BLE path (optional, requires: pip install frame-sdk):
    - CollectiveOS connects directly to Frame glasses via BLE
    - Enables two-way: capture photos, stream frames, show display text
    - Activated when FRAME_BLE_DEVICE env var is set and frame-sdk is installed

Display constraints:
    - Frame shows white text on transparent background in the user's field of view
    - Effective area: approx 640×400 logical display pixels
    - Practical limit: ~25 chars per line, ~6 lines max visible at once
    - format_for_frame() enforces these limits

Nav agent integration:
    - When the wearable WebSocket stream triggers navigate_computer, the stored
      Frame camera frame is passed as first_person_frame= so the agent can see
      what the user physically sees (e.g. a restaurant sign → book a table there)
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
from typing import Optional

log = logging.getLogger(__name__)

_FRAME_LINE_WIDTH  = int(os.getenv("FRAME_LINE_WIDTH", "25"))  # chars per display line
_FRAME_MAX_LINES   = int(os.getenv("FRAME_MAX_LINES",  "6"))   # lines visible at once
_FRAME_BLE_DEVICE  = os.getenv("FRAME_BLE_DEVICE", "")         # empty = BLE disabled


# ── Display formatting ────────────────────────────────────────────────────────

def format_for_frame(text: str) -> str:
    """
    Format an agent reply for Frame's in-lens display.

    Strips markdown, truncates to fit the display area, and wraps to
    _FRAME_LINE_WIDTH chars × _FRAME_MAX_LINES lines.

    Returns a plain-text string ready to send to the glasses.
    """
    # Strip markdown: bold, italic, code, links, headers
    clean = re.sub(r"\*{1,2}|`{1,3}|#{1,6}\s?", "", text)
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)  # [label](url) → label
    clean = re.sub(r"\n{2,}", "\n", clean).strip()

    max_chars = _FRAME_LINE_WIDTH * _FRAME_MAX_LINES
    if len(clean) > max_chars:
        clean = clean[:max_chars - 1] + "…"

    lines = textwrap.wrap(clean, width=_FRAME_LINE_WIDTH)
    return "\n".join(lines[:_FRAME_MAX_LINES])


# ── Direct BLE path (optional frame-sdk) ─────────────────────────────────────

class FrameBLEConnector:
    """
    Direct BLE connection to Frame glasses via the frame-sdk library.
    Instantiate once at startup; call connect() before use.

    Requires: pip install frame-sdk
    Requires: FRAME_BLE_DEVICE env var set to the glasses' BLE address or name.
    """

    def __init__(self) -> None:
        self._frame: object = None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to Frame glasses. Returns True on success."""
        if not _FRAME_BLE_DEVICE:
            log.debug("Frame BLE: FRAME_BLE_DEVICE not set — BLE path disabled.")
            return False
        try:
            from frame_sdk import Frame  # type: ignore[import]
            self._frame = Frame()
            await self._frame.connect()  # type: ignore[attr-defined]
            self._connected = True
            log.info("Frame BLE connected to %s", _FRAME_BLE_DEVICE)
            return True
        except ImportError:
            log.debug("Frame BLE: frame-sdk not installed (pip install frame-sdk).")
            return False
        except Exception as exc:
            log.warning("Frame BLE connect failed: %s", exc)
            return False

    async def disconnect(self) -> None:
        if self._connected and self._frame:
            try:
                await self._frame.disconnect()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._connected = False

    async def show_text(self, text: str) -> bool:
        """Display formatted text on Frame's in-lens screen."""
        if not self._connected or not self._frame:
            return False
        formatted = format_for_frame(text)
        try:
            await self._frame.display.show_text(formatted)  # type: ignore[attr-defined]
            return True
        except Exception as exc:
            log.warning("Frame display error: %s", exc)
            return False

    async def capture_photo(self) -> Optional[bytes]:
        """Capture a JPEG photo from Frame's camera. Returns bytes or None."""
        if not self._connected or not self._frame:
            return None
        try:
            photo = await self._frame.camera.take_photo()  # type: ignore[attr-defined]
            return bytes(photo) if photo else None
        except Exception as exc:
            log.warning("Frame camera error: %s", exc)
            return None

    async def clear_display(self) -> None:
        if self._connected and self._frame:
            try:
                await self._frame.display.clear()  # type: ignore[attr-defined]
            except Exception:
                pass

    @property
    def connected(self) -> bool:
        return self._connected


# ── Singleton BLE connector ───────────────────────────────────────────────────

_ble: Optional[FrameBLEConnector] = None


def get_ble_connector() -> FrameBLEConnector:
    global _ble
    if _ble is None:
        _ble = FrameBLEConnector()
    return _ble
