"""
Home Assistant connector.

Read device states and control switches/lights via the HA REST API.

Auth: long-lived access token.
  1. In Home Assistant: Profile → Security → Long-Lived Access Tokens → Create.
  2. Add to .env:
       HASS_URL=http://homeassistant.local:8123
       HASS_TOKEN=your-long-lived-token

Safety rules (from CLAUDE.md):
  - Never remotely START heating appliances (microwave, cooktop, washer).
    Switch and monitor only.
  - The agent must confirm with the user before calling control_device().
"""

import os
import requests

# Domains that may never be turned ON remotely — monitor/off only.
_HEATING_DOMAINS = {"climate"}
_HEATING_KEYWORDS = {"microwave", "cooktop", "washer", "oven", "stove", "dryer"}


def _session() -> requests.Session:
    url   = os.environ.get("HASS_URL", "").rstrip("/")
    token = os.environ.get("HASS_TOKEN", "")
    if not url or not token:
        raise RuntimeError(
            "HASS_URL and HASS_TOKEN must be set in your .env file.\n"
            "Get a token from: Home Assistant → Profile → Security → "
            "Long-Lived Access Tokens."
        )
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    s.base_url = url  # type: ignore[attr-defined]
    return s


def _get(path: str) -> dict | list:
    s = _session()
    resp = s.get(f"{s.base_url}{path}")
    resp.raise_for_status()
    return resp.json()


def _post(path: str, payload: dict) -> dict:
    s = _session()
    resp = s.post(f"{s.base_url}{path}", json=payload)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_devices(domain: str = "") -> str:
    """
    List Home Assistant entities and their current states.

    - domain: optional filter, e.g. 'light', 'switch', 'sensor', 'binary_sensor',
              'climate', 'media_player'. Leave blank for all entities.
    """
    states = _get("/api/states")

    if domain:
        states = [s for s in states if s["entity_id"].startswith(f"{domain}.")]

    if not states:
        return f"No entities found{' for domain: ' + domain if domain else ''}."

    lines = []
    for s in sorted(states, key=lambda x: x["entity_id"]):
        attrs  = s.get("attributes", {})
        name   = attrs.get("friendly_name", s["entity_id"])
        state  = s["state"]
        extra  = ""
        if "brightness" in attrs:
            pct = round(attrs["brightness"] / 255 * 100)
            extra = f" ({pct}% brightness)"
        elif "temperature" in attrs:
            extra = f" ({attrs['temperature']}°)"
        lines.append(f"- {s['entity_id']}  [{name}]  →  {state}{extra}")

    return "\n".join(lines)


def get_device_state(entity_id: str) -> str:
    """Get the full state and attributes of a single Home Assistant entity."""
    data = _get(f"/api/states/{entity_id}")
    attrs = data.get("attributes", {})
    name  = attrs.get("friendly_name", entity_id)
    lines = [f"Entity: {entity_id}  ({name})", f"State:  {data['state']}"]
    for k, v in attrs.items():
        if k != "friendly_name":
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Control  (confirm with user before calling)
# ---------------------------------------------------------------------------

def control_device(entity_id: str, action: str) -> str:
    """
    Control a Home Assistant entity.

    - entity_id: e.g. 'light.living_room', 'switch.fan'
    - action:    'turn_on' or 'turn_off'

    Safety: heating appliances (microwave, cooktop, washer, oven, stove,
    dryer, climate entities) may only be turned OFF, never ON.
    """
    domain = entity_id.split(".")[0]
    name_lower = entity_id.lower()

    is_heating = domain in _HEATING_DOMAINS or any(
        kw in name_lower for kw in _HEATING_KEYWORDS
    )
    if is_heating and action == "turn_on":
        return (
            f"Refused: '{entity_id}' looks like a heating appliance. "
            "Remote start is blocked for safety. You can turn it OFF or monitor its state."
        )

    if action not in {"turn_on", "turn_off"}:
        return f"Unknown action '{action}'. Use 'turn_on' or 'turn_off'."

    _post(f"/api/services/{domain}/{action}", {"entity_id": entity_id})
    friendly = action.replace("_", " ")
    return f"Done — {friendly}: {entity_id}"
