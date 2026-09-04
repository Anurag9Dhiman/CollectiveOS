"""
Screen capture connector — screenshot + vision analysis.

Routes through VisualOS (Lens OS) when LENS_URL is configured, falling back
to Gemini Vision when it is not.  The image is held in memory only and deleted
immediately after analysis — it is never written anywhere permanently.

VisualOS API: POST {LENS_URL}/analyze  (multipart/form-data)
  Headers: X-API-Key: {LENS_API_KEY}
  Fields:  image (file), user_id (str)
  Returns: {"card": {"card_type": "normal"|"fallback", "headline": ..., "body": ...}, ...}

Privacy note: the screenshot is sent to an external API for analysis.
Disable this connector in Settings if you don't want that.
"""

import os
import platform
import subprocess
import tempfile

_VISION_MODEL = os.environ.get("VISION_MODEL", "gemini-3.6-flash")
_MAX_PX = 1280  # resize longest dimension before sending


def capture_screen(question: str = "") -> str:
    """
    Take a screenshot of the current Mac screen and return a vision analysis.

    question: Specific question about the screen, e.g. "What does the error say?"
              Leave blank for a general description.
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

        subprocess.run(["sips", "-Z", str(_MAX_PX), tmp], capture_output=True, timeout=10)

        with open(tmp, "rb") as fh:
            image_bytes = fh.read()
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    lens_url = os.environ.get("LENS_URL", "").rstrip("/")
    if lens_url:
        return _analyze_with_visualos(image_bytes, question, lens_url)
    return _analyze_with_gemini(image_bytes, question)


# ---------------------------------------------------------------------------
# VisualOS (Lens OS) backend
# ---------------------------------------------------------------------------

def _analyze_with_visualos(image_bytes: bytes, question: str, base_url: str) -> str:
    """POST the screenshot to VisualOS /analyze and format the card response."""
    import io
    import json
    import urllib.request

    api_key = os.environ.get("LENS_API_KEY", "")
    boundary = "----CollectiveOSBoundary"
    filename = "screen.png"

    # Build multipart/form-data body manually (no external deps)
    parts: list[bytes] = []

    def field(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()

    parts.append(field("user_id", "collectiveos"))
    if question.strip():
        parts.append(field("query", question.strip()))

    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n".encode()
        + image_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        req = urllib.request.Request(
            f"{base_url}/analyze",
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        card = data.get("card", {})
        card_type = card.get("card_type", "fallback")

        if card_type == "normal":
            headline = card.get("headline", "")
            body_text = card.get("body", "")
            return f"{headline}\n\n{body_text}".strip()
        else:
            # fallback card
            headline = card.get("headline", "")
            observation = card.get("observation", "")
            suggestion = card.get("suggestion", "")
            parts_out = [p for p in [headline, observation, suggestion] if p]
            return "\n\n".join(parts_out) or "VisualOS returned an empty response."

    except Exception as e:
        # VisualOS unreachable — fall back to Gemini
        fallback = _analyze_with_gemini(image_bytes, question)
        return f"[VisualOS unavailable: {e}. Gemini fallback:]\n\n{fallback}"


# ---------------------------------------------------------------------------
# Gemini Vision fallback
# ---------------------------------------------------------------------------

def _analyze_with_gemini(image_bytes: bytes, question: str) -> str:
    """Analyse the screenshot directly with Gemini Vision."""
    from google import genai
    from google.genai import types as _gtypes

    prompt = question.strip() or (
        "Describe what you see on this screen in detail. "
        "Include the app(s) visible, any text content, and anything notable."
    )
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model=_VISION_MODEL,
            contents=[
                _gtypes.Part(
                    inline_data=_gtypes.Blob(mime_type="image/png", data=image_bytes)
                ),
                prompt,
            ],
        )
        return response.text
    except Exception as e:
        return f"Vision analysis error: {e}"
