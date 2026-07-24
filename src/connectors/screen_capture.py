"""
Screen capture connector — take a screenshot and analyse it with Claude vision.

Uses macOS `screencapture` to grab the current screen, `sips` to resize it
to a token-efficient size, then passes it to the Anthropic vision API.

The image is held in memory only and deleted immediately after analysis —
it is never written anywhere permanently.

Privacy note: the screenshot is sent to Anthropic's API for analysis.
Disable this connector in Settings if you don't want that.
"""

import base64
import os
import platform
import subprocess
import tempfile

from anthropic import Anthropic

_VISION_MODEL = "claude-sonnet-4-6"
_MAX_PX = 1280  # resize longest dimension to this before sending


def capture_screen(question: str = "") -> str:
    """
    Take a screenshot of the current Mac screen and return a vision analysis.

    question: Specific question to answer about screen contents, e.g.
              "What does the error message say?" or "What app is open?"
              Leave blank for a general description of what's visible.

    The image is resized to max 1280 px on the longest side before being
    sent to the vision model to keep token cost reasonable.
    """
    if platform.system() != "Darwin":
        return "Screen capture only works on macOS."

    tmp = tempfile.mktemp(suffix=".png")
    try:
        # Capture screen silently (-x suppresses the shutter sound)
        result = subprocess.run(
            ["screencapture", "-x", tmp],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return f"Screenshot failed: {result.stderr.strip()}"

        if not os.path.exists(tmp):
            return "Screenshot was not created — screencapture returned no file."

        # Resize in-place with sips (built-in macOS tool, no dependencies)
        subprocess.run(
            ["sips", "-Z", str(_MAX_PX), tmp],
            capture_output=True, timeout=10,
        )

        with open(tmp, "rb") as fh:
            image_b64 = base64.standard_b64encode(fh.read()).decode("utf-8")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    prompt = question.strip() or (
        "Describe what you see on this screen in detail. "
        "Include the app(s) visible, any text content, and anything notable."
    )

    try:
        client = Anthropic()
        resp = client.messages.create(
            model=_VISION_MODEL,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return resp.content[0].text
    except Exception as e:
        return f"Vision analysis error: {e}"
