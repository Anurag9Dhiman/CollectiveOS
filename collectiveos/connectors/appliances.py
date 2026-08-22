"""
Smart appliances connector — multi-brand: Samsung SmartThings, LG ThinQ.

Select your brand in Settings ⚙ → Smart Appliances → Brand dropdown (saved to DB).
Fallback: set APPLIANCE_BRAND=smartthings|lgthinq in .env.

SmartThings env vars:  SMARTTHINGS_TOKEN (personal access token)
LG ThinQ env vars:     LG_USERNAME, LG_PASSWORD, LG_COUNTRY (e.g. US)

Safety: heating appliances (washer, dryer, oven, microwave, cooktop) cannot be
remotely started — 'off' is the only allowed command for those devices.
"""

import os

import requests

_ST_BASE = "https://api.smartthings.com/v1"

# SmartThings capability IDs that identify heating appliances
_HEATING_CAPS = {
    "washerOperatingState", "dryerOperatingState",
    "ovenOperatingState",   "dishwasherOperatingState",
}

# Keywords in device label that also trigger the heating-appliance block
_HEATING_WORDS = {"washer", "dryer", "oven", "microwave", "cooktop", "range", "dishwasher"}


def _get_brand() -> str:
    try:
        from collectiveos.permissions import get_config
        brand = get_config("appliances").get("brand", "")
        if brand:
            return brand.lower()
    except Exception:
        pass
    return os.environ.get("APPLIANCE_BRAND", "").lower()


# ---------------------------------------------------------------------------
# SmartThings
# ---------------------------------------------------------------------------

def _sth() -> dict:
    return {
        "Authorization": f"Bearer {os.environ.get('SMARTTHINGS_TOKEN', '')}",
        "Content-Type": "application/json",
    }


def _st_check() -> str | None:
    if not os.environ.get("SMARTTHINGS_TOKEN"):
        return "SMARTTHINGS_TOKEN is not set. See .env.example."
    return None


def _is_heating(device: dict) -> bool:
    label = (device.get("label") or "").lower()
    if any(kw in label for kw in _HEATING_WORDS):
        return True
    caps = {
        c.get("id", "")
        for comp in device.get("components", [])
        for c in comp.get("capabilities", [])
    }
    return bool(_HEATING_CAPS & caps)


def _st_list() -> str:
    err = _st_check()
    if err:
        return err
    try:
        resp = requests.get(f"{_ST_BASE}/devices", headers=_sth(), timeout=15)
        resp.raise_for_status()
        devices = resp.json().get("items", [])
        if not devices:
            return "No SmartThings devices found. Make sure the integration is connected."
        lines = ["SmartThings devices:"]
        for d in devices:
            label = d.get("label") or d.get("name", "Unknown")
            dtype = d.get("deviceTypeName") or ""
            did   = d.get("deviceId", "")
            note  = " [monitor only — heating appliance]" if _is_heating(d) else ""
            lines.append(f"  {label}  [{dtype}]  ID: {did}{note}")
        return "\n".join(lines)
    except Exception as exc:
        return f"SmartThings list error: {exc}"


def _st_status(device_id: str) -> str:
    err = _st_check()
    if err:
        return err
    try:
        resp = requests.get(
            f"{_ST_BASE}/devices/{device_id}/status",
            headers=_sth(), timeout=15,
        )
        resp.raise_for_status()
        components = resp.json().get("components", {})
        lines = [f"Status for {device_id}:"]
        for comp_id, capabilities in components.items():
            for cap_id, attrs in capabilities.items():
                for attr, val_obj in attrs.items():
                    value = val_obj.get("value")
                    unit  = val_obj.get("unit", "")
                    unit_str = f" {unit}" if unit else ""
                    lines.append(f"  [{cap_id}] {attr}: {value}{unit_str}")
        return "\n".join(lines) if len(lines) > 1 else f"No status data for {device_id}."
    except Exception as exc:
        return f"SmartThings status error: {exc}"


def _st_control(device_id: str, command: str) -> str:
    err = _st_check()
    if err:
        return err
    # Fetch device info for the safety check
    try:
        dev_resp = requests.get(f"{_ST_BASE}/devices/{device_id}", headers=_sth(), timeout=15)
        device = dev_resp.json() if dev_resp.ok else {}
    except Exception:
        device = {}

    if _is_heating(device) and command.lower() != "off":
        label = device.get("label", device_id)
        return (
            f"Safety block: '{label}' is a heating appliance. "
            "Remote start is not allowed — only 'off' commands are permitted."
        )

    cmd = command.lower()
    body = {
        "commands": [{
            "component":  "main",
            "capability": "switch",
            "command":    "on" if cmd == "on" else "off",
        }]
    }
    try:
        resp = requests.post(
            f"{_ST_BASE}/devices/{device_id}/commands",
            headers=_sth(), json=body, timeout=15,
        )
        resp.raise_for_status()
        return f"Sent '{command}' to device {device_id}."
    except Exception as exc:
        return f"SmartThings control error: {exc}"


# ---------------------------------------------------------------------------
# LG ThinQ — stub with setup guidance
# ---------------------------------------------------------------------------

def _lg_stub() -> str:
    return (
        "LG ThinQ integration requires thinq2-python. "
        "pip install thinq2-python, then set LG_USERNAME, LG_PASSWORD, LG_COUNTRY (e.g. US) in .env. "
        "Full implementation coming soon."
    )


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

def appliances_list() -> str:
    """List all smart appliances with their IDs. Heating appliances are marked monitor-only."""
    brand = _get_brand()
    if brand == "smartthings":
        return _st_list()
    if brand == "lgthinq":
        return _lg_stub()
    return "No appliance brand selected. Open Settings ⚙ → Smart Appliances and choose your brand."


def appliances_get_status(device_id: str) -> str:
    """Get the detailed status of a single appliance by its device ID."""
    brand = _get_brand()
    if brand == "smartthings":
        return _st_status(device_id.strip())
    if brand == "lgthinq":
        return _lg_stub()
    return "No appliance brand selected."


def appliances_control(device_id: str, command: str) -> str:
    """
    Turn an appliance on or off. command: 'on' or 'off'.
    Heating appliances (washer, dryer, oven, microwave, cooktop) are blocked from 'on'.
    Always confirm with the user before calling.
    """
    brand = _get_brand()
    if brand == "smartthings":
        return _st_control(device_id.strip(), command)
    if brand == "lgthinq":
        return _lg_stub()
    return "No appliance brand selected."
