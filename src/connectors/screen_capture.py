"""
Screen capture connector — take a screenshot and analyse it with Gemini Vision.

Uses macOS `screencapture` to grab the current screen, `sips` to resize it
to a token-efficient size, then passes the raw bytes to the Gemini vision API.

The image is held in memory only and deleted immediately after analysis —
it is never written anywhere permanently.

Privacy note: the screenshot is sent to Google's API for analysis.
Disable this connector in Settings if you don't want that.
"""

import base64
import os
import platform
import subprocess
import tempfile

import google.generativeai as genai

_VISION_MODEL = "gemini-2.0-flash"
_MAX_PX = 1280  # resize longest dimension to this before sending


def capture_screen(question: str = "") -> str:
    """
    Take a screenshot of the current Mac screen and return a vision analysis.

    question: Specific question to answer about screen contents, e.g.
              "What does the error message say?" or "What app is open?"
              Leave blank for a general description of what's visible.
    """
    if platform.system() != "Darwin":
        return "Screen capture only works on macOS."

    tmp = tempfile.mktemp(suffix=".png")
    try:
        result = subprocess.run(
            ["screencapture", "-x", tmp],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return f"Screenshot failed: {result.stderr.strip()}"

        if not os.path.exists(tmp):
            return "Screenshot was not created — screencapture returned no file."

        subprocess.run(
            ["sips", "-Z", str(_MAX_PX), tmp],
            capture_output=True, timeout=10,
        )

        with open(tmp, "rb") as fh:
            image_bytes = fh.read()
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    prompt = question.strip() or (
        "Describe what you see on this screen in detail. "
        "Include the app(s) visible, any text content, and anything notable."
    )

    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(_VISION_MODEL)
        response = model.generate_content([
            genai.protos.Part(
                inline_data=genai.protos.Blob(
                    mime_type="image/png",
                    data=image_bytes,
                )
            ),
            prompt,
        ])
        return response.text
    except Exception as e:
        return f"Vision analysis error: {e}"
