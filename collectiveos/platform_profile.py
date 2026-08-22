"""
Platform profile — detects the device type and which connectors make sense on it.

Called by `collectiveos info` and the init flow. Connectors that are
platform-specific (e.g. iMessage is Mac-only) are marked unavailable on
unsupported platforms so they are silently excluded from the tool list.
"""

import os
import platform
import sys


def detect() -> dict:
    """Return a dict describing the current device."""
    system  = platform.system()          # Darwin, Linux, Windows
    machine = platform.machine()         # arm64, x86_64, aarch64
    node    = platform.node()

    device_type = "unknown"
    if system == "Darwin":
        device_type = "mac"
    elif system == "Linux":
        if _is_raspberry_pi():
            device_type = "raspberry_pi"
        elif os.environ.get("container") or os.path.exists("/.dockerenv"):
            device_type = "container"
        else:
            device_type = "linux"
    elif system == "Windows":
        device_type = "windows"

    return {
        "os":          system,
        "arch":        machine,
        "python":      sys.version.split()[0],
        "hostname":    node,
        "device_type": device_type,
        "is_mac":      system == "Darwin",
        "is_linux":    system == "Linux",
        "is_arm":      "arm" in machine.lower() or "aarch" in machine.lower(),
        "is_pi":       _is_raspberry_pi(),
        "is_container": os.environ.get("container") == "podman" or os.path.exists("/.dockerenv"),
    }


def _is_raspberry_pi() -> bool:
    try:
        with open("/proc/cpuinfo") as f:
            return "Raspberry Pi" in f.read()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Connector availability matrix
# Each entry: (connector_name, available_fn, note_when_unavailable)
# ---------------------------------------------------------------------------

_CONNECTORS = [
    # Always available
    ("memory",         lambda p: True,        ""),
    ("web_search",     lambda p: True,        ""),
    ("ai_models",      lambda p: True,        ""),
    ("notion",         lambda p: True,        ""),
    ("github",         lambda p: True,        ""),
    ("telegram",       lambda p: True,        ""),
    ("slack",          lambda p: True,        ""),
    ("finance",        lambda p: True,        ""),
    ("health",         lambda p: True,        ""),

    # Google services — available everywhere with OAuth
    ("google_calendar", lambda p: True,       ""),
    ("gmail",           lambda p: True,       ""),
    ("google_drive",    lambda p: True,       ""),
    ("todoist",         lambda p: True,       ""),

    # Mac-only
    ("imessage",      lambda p: p["is_mac"],  "macOS only"),
    ("mac_system",    lambda p: p["is_mac"],  "macOS only"),
    ("screen_capture",lambda p: p["is_mac"],  "macOS only"),
    ("contacts",      lambda p: p["is_mac"],  "macOS only"),
    ("reminders",     lambda p: p["is_mac"],  "macOS only"),
    ("notes",         lambda p: p["is_mac"],  "macOS only"),
    ("browser",       lambda p: p["is_mac"],  "macOS only"),
    ("clipboard",     lambda p: p["is_mac"],  "macOS only"),
    ("spotify",       lambda p: p["is_mac"],  "macOS only"),

    # Smart home — available on Mac + Pi + Linux (not Windows)
    ("home_assistant", lambda p: not p["os"] == "Windows", ""),
    ("appliances",     lambda p: not p["os"] == "Windows", ""),

    # Car — available anywhere (cloud API)
    ("car",            lambda p: True,        ""),

    # Local files — available everywhere
    ("local_files",    lambda p: True,        ""),
]


def enabled_connectors(profile: dict) -> list[tuple[str, bool, str]]:
    """
    Return list of (connector_name, is_enabled, note) for the given profile.
    """
    result = []
    for name, check, note in _CONNECTORS:
        try:
            enabled = check(profile)
        except Exception:
            enabled = False
        result.append((name, enabled, note if not enabled else ""))
    return result


def enabled_tool_names(profile: dict | None = None) -> set[str]:
    """
    Return the set of tool names that should be active on this device.
    Used by the router and the agent loop to filter the tool list at startup.
    """
    if profile is None:
        profile = detect()

    from collectiveos.permissions import CONNECTOR_TOOLS
    enabled = {name for name, ok, _ in enabled_connectors(profile) if ok}
    tools: set[str] = set()
    for connector in enabled:
        tools.update(CONNECTOR_TOOLS.get(connector, []))
    return tools
