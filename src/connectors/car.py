"""
Car connector — multi-brand: Tesla, BMW/MINI, Ford/Lincoln.

Select your brand in Settings ⚙ → Car → Brand dropdown (saved to DB).
Fallback: set CAR_BRAND=tesla|bmw|ford in .env.

Tesla env vars:  TESLA_TOKEN (Bearer access token)
                 TESLA_VEHICLE_ID (optional, defaults to first vehicle)
                 TESLA_FLEET_BASE_URL (optional, default NA region)
BMW env vars:    BMW_USERNAME, BMW_PASSWORD, BMW_VIN
Ford env vars:   FORD_USERNAME, FORD_PASSWORD
"""

import os

import requests

_TESLA_BASE = os.environ.get(
    "TESLA_FLEET_BASE_URL", "https://fleet-api.prd.na.vn.cloud.tesla.com/api/1"
)
_tesla_cached_id: str | None = None  # cached after first lookup


def _get_brand() -> str:
    try:
        from src.permissions import get_config
        brand = get_config("car").get("brand", "")
        if brand:
            return brand.lower()
    except Exception:
        pass
    return os.environ.get("CAR_BRAND", "").lower()


# ---------------------------------------------------------------------------
# Tesla
# ---------------------------------------------------------------------------

def _th() -> dict:
    return {"Authorization": f"Bearer {os.environ.get('TESLA_TOKEN', '')}"}


def _tesla_vid() -> tuple[str, str | None]:
    global _tesla_cached_id
    if _tesla_cached_id:
        return _tesla_cached_id, None
    if not os.environ.get("TESLA_TOKEN"):
        return "", "TESLA_TOKEN is not set. See .env.example."
    env_vid = os.environ.get("TESLA_VEHICLE_ID", "")
    if env_vid:
        _tesla_cached_id = env_vid
        return env_vid, None
    try:
        resp = requests.get(f"{_TESLA_BASE}/vehicles", headers=_th(), timeout=15)
        resp.raise_for_status()
        vehicles = resp.json().get("response", [])
        if not vehicles:
            return "", "No vehicles found on this Tesla account."
        _tesla_cached_id = str(vehicles[0]["id"])
        return _tesla_cached_id, None
    except Exception as exc:
        return "", f"Tesla vehicle lookup error: {exc}"


def _tesla_status() -> str:
    vid, err = _tesla_vid()
    if err:
        return err
    try:
        resp = requests.get(
            f"{_TESLA_BASE}/vehicles/{vid}/vehicle_data",
            headers=_th(),
            params={"endpoints": "charge_state,climate_state,drive_state,vehicle_state"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json().get("response", {})
        if not data:
            return (
                "Vehicle appears to be asleep. Open the Tesla app to wake it, "
                "then try again."
            )
        name = data.get("display_name", "Your Tesla")
        vs = data.get("vehicle_state", {})
        cs = data.get("charge_state", {})
        cl = data.get("climate_state", {})
        ds = data.get("drive_state", {})
        locked = "🔒 locked" if vs.get("locked") else "🔓 unlocked"
        inside = cl.get("inside_temp")
        outside = cl.get("outside_temp")
        temp_str = f"{inside:.1f}°C inside / {outside:.1f}°C outside" if inside else "—"
        return (
            f"{name}  {locked}\n"
            f"  Battery: {cs.get('battery_level', '—')}%  "
            f"Range: {cs.get('battery_range', '—')} mi  "
            f"Charging: {cs.get('charging_state', '—')}\n"
            f"  Climate: {'on' if cl.get('is_climate_on') else 'off'}  "
            f"Temps: {temp_str}\n"
            f"  Speed: {ds.get('speed') or 0} mph  "
            f"Location: {ds.get('latitude', '—')}, {ds.get('longitude', '—')}"
        )
    except Exception as exc:
        return f"Tesla status error: {exc}"


def _tesla_lock(action: str) -> str:
    vid, err = _tesla_vid()
    if err:
        return err
    cmd = "door_lock" if action == "lock" else "door_unlock"
    try:
        resp = requests.post(
            f"{_TESLA_BASE}/vehicles/{vid}/command/{cmd}",
            headers=_th(), json={}, timeout=20,
        )
        result = resp.json().get("response", {})
        if result.get("result"):
            return f"Car {action}ed successfully."
        return f"Command failed: {result.get('reason', 'unknown error')}"
    except Exception as exc:
        return f"Tesla lock error: {exc}"


def _tesla_climate(action: str, temp_f: float = 70.0) -> str:
    vid, err = _tesla_vid()
    if err:
        return err
    try:
        if action == "start":
            temp_c = (temp_f - 32) * 5 / 9
            requests.post(
                f"{_TESLA_BASE}/vehicles/{vid}/command/set_temps",
                headers=_th(),
                json={"driver_temp": temp_c, "passenger_temp": temp_c},
                timeout=20,
            )
            resp = requests.post(
                f"{_TESLA_BASE}/vehicles/{vid}/command/auto_conditioning_start",
                headers=_th(), json={}, timeout=20,
            )
        else:
            resp = requests.post(
                f"{_TESLA_BASE}/vehicles/{vid}/command/auto_conditioning_stop",
                headers=_th(), json={}, timeout=20,
            )
        result = resp.json().get("response", {})
        if result.get("result"):
            return f"Climate {'started at ' + str(temp_f) + '°F' if action == 'start' else 'stopped'}."
        return f"Climate command failed: {result.get('reason', 'unknown')}"
    except Exception as exc:
        return f"Tesla climate error: {exc}"


# ---------------------------------------------------------------------------
# BMW / Ford — stubs with setup guidance
# ---------------------------------------------------------------------------

def _bmw_stub() -> str:
    return (
        "BMW/MINI integration requires bimmer-connected. "
        "pip install bimmer-connected, then set BMW_USERNAME, BMW_PASSWORD, BMW_VIN in .env. "
        "Full implementation coming soon."
    )


def _ford_stub() -> str:
    return (
        "Ford/Lincoln integration requires fordpass-python. "
        "pip install fordpass, then set FORD_USERNAME, FORD_PASSWORD in .env. "
        "Full implementation coming soon."
    )


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

def car_get_status() -> str:
    """Get car status — battery/charge, range, lock state, climate, location."""
    brand = _get_brand()
    if brand == "tesla":
        return _tesla_status()
    if brand == "bmw":
        return _bmw_stub()
    if brand == "ford":
        return _ford_stub()
    return "No car brand selected. Open Settings ⚙ → Car and choose your brand."


def car_lock(action: str) -> str:
    """Lock or unlock the car. action: 'lock' or 'unlock'. Always confirm first."""
    brand = _get_brand()
    if brand == "tesla":
        return _tesla_lock(action)
    if brand in ("bmw", "ford"):
        return _bmw_stub() if brand == "bmw" else _ford_stub()
    return "No car brand selected."


def car_climate(action: str, temp_f: float = 70.0) -> str:
    """Start or stop climate pre-conditioning. action: 'start' or 'stop'. Always confirm first."""
    brand = _get_brand()
    if brand == "tesla":
        return _tesla_climate(action, temp_f)
    if brand in ("bmw", "ford"):
        return _bmw_stub() if brand == "bmw" else _ford_stub()
    return "No car brand selected."
