"""
Browser connector — read tabs and open URLs on Mac via osascript.

Tries Safari first, then Google Chrome. Both must have Automation permission
granted in System Settings → Privacy & Security → Automation (macOS will
prompt once on first use).

Read  tools: browser_get_active_tab, browser_list_tabs
Write tool : browser_open_url  (always confirm with user before calling)
"""

import platform
import re
import subprocess


def _require_macos() -> str | None:
    if platform.system() != "Darwin":
        return "Browser tools only work on macOS."
    return None


def _run_script(script: str, timeout: int = 10) -> tuple[str, str]:
    """Run an AppleScript and return (stdout, stderr)."""
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout.strip(), result.stderr.strip()


def _is_running(app: str) -> bool:
    out, _ = _run_script(
        f'tell application "System Events" to '
        f'(name of processes) contains "{app}"'
    )
    return out.lower() == "true"


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

def get_active_tab() -> str:
    """
    Return the title and URL of the currently active browser tab.
    Checks Safari first, then Google Chrome. Returns whichever is frontmost,
    or tries both if neither is the active application.
    """
    err = _require_macos()
    if err:
        return err

    # Try the frontmost browser first
    front_app, _ = _run_script(
        'tell application "System Events" to '
        'get name of first application process whose frontmost is true'
    )

    browsers = []
    if "Safari" in front_app:
        browsers = ["Safari", "Google Chrome"]
    elif "Chrome" in front_app or "Chromium" in front_app:
        browsers = ["Google Chrome", "Safari"]
    else:
        browsers = ["Safari", "Google Chrome"]

    for browser in browsers:
        if not _is_running(browser):
            continue

        if browser == "Safari":
            script = (
                'tell application "Safari"\n'
                '  if (count of windows) > 0 then\n'
                '    set t to current tab of window 1\n'
                '    return (name of t) & "\n" & (URL of t)\n'
                '  end if\n'
                'end tell'
            )
        else:
            script = (
                'tell application "Google Chrome"\n'
                '  if (count of windows) > 0 then\n'
                '    set t to active tab of window 1\n'
                '    return (title of t) & "\n" & (URL of t)\n'
                '  end if\n'
                'end tell'
            )

        out, err_msg = _run_script(script)
        if out and "\n" in out:
            title, url = out.split("\n", 1)
            return f"Browser: {browser}\nTitle:   {title.strip()}\nURL:     {url.strip()}"
        if out:
            return f"Browser: {browser}\nTab:     {out}"

    return "No supported browser (Safari or Chrome) is currently open."


def list_tabs() -> str:
    """
    List all open tabs across all windows in Safari and Google Chrome.
    Returns tab titles and URLs grouped by browser and window.
    """
    err = _require_macos()
    if err:
        return err

    results = []

    # Safari
    if _is_running("Safari"):
        script = (
            'tell application "Safari"\n'
            '  set output to ""\n'
            '  repeat with w from 1 to count of windows\n'
            '    set output to output & "Window " & w & ":\\n"\n'
            '    repeat with t in tabs of window w\n'
            '      set output to output & "  " & (name of t) & "\\n"\n'
            '      set output to output & "  " & (URL of t) & "\\n"\n'
            '    end repeat\n'
            '  end repeat\n'
            '  return output\n'
            'end tell'
        )
        out, _ = _run_script(script, timeout=15)
        if out:
            results.append("=== Safari ===\n" + out)

    # Chrome
    if _is_running("Google Chrome"):
        script = (
            'tell application "Google Chrome"\n'
            '  set output to ""\n'
            '  repeat with w from 1 to count of windows\n'
            '    set output to output & "Window " & w & ":\\n"\n'
            '    repeat with t in tabs of window w\n'
            '      set output to output & "  " & (title of t) & "\\n"\n'
            '      set output to output & "  " & (URL of t) & "\\n"\n'
            '    end repeat\n'
            '  end repeat\n'
            '  return output\n'
            'end tell'
        )
        out, _ = _run_script(script, timeout=15)
        if out:
            results.append("=== Google Chrome ===\n" + out)

    if not results:
        return "No supported browser (Safari or Chrome) is currently open."

    return "\n".join(results)


# ---------------------------------------------------------------------------
# Write tool
# ---------------------------------------------------------------------------

_ALLOWED_SCHEMES = re.compile(r'^https?://', re.IGNORECASE)


def open_url(url: str) -> str:
    """
    Open a URL in the default browser.

    Only http:// and https:// URLs are accepted — file://, javascript:,
    and data: are blocked. Always confirm with the user before calling.

    url: The full URL to open, e.g. 'https://example.com'.
    """
    err = _require_macos()
    if err:
        return err

    url = url.strip()
    if not _ALLOWED_SCHEMES.match(url):
        return (
            "Only http:// and https:// URLs can be opened. "
            f"Rejected: {url!r}"
        )

    result = subprocess.run(
        ["open", url],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return f"Failed to open URL: {result.stderr.strip()}"
    return f"Opened in browser: {url}"
