"""
Spotify connector — read playback state and control music.

Works with any Spotify Connect device: phone, laptop, speaker, car, TV.

Setup (one-time):
  1. Go to https://developer.spotify.com/dashboard and create an app.
  2. In the app settings, add redirect URI: http://localhost:8888/callback
  3. Copy Client ID and Client Secret into .env:
       SPOTIFY_CLIENT_ID=your-client-id
       SPOTIFY_CLIENT_SECRET=your-client-secret
  4. On first run the assistant will open a browser for Spotify login.
     After you approve, a .spotify_cache file is written and reused.

Scopes used:
  user-read-playback-state    — see current track and device
  user-modify-playback-state  — play, pause, skip, volume, queue
  user-read-currently-playing — currently playing track details
"""

import os
from functools import lru_cache

import spotipy
from spotipy.oauth2 import SpotifyOAuth

_SCOPES = " ".join([
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
])

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.path.join(_HERE, "..", "..", ".spotify_cache")


@lru_cache(maxsize=1)
def _client() -> spotipy.Spotify:
    client_id     = os.environ.get("SPOTIFY_CLIENT_ID", "")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    redirect_uri  = os.environ.get("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

    if not client_id or not client_secret:
        raise RuntimeError(
            "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in .env.\n"
            "Create a free app at https://developer.spotify.com/dashboard"
        )

    auth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=_SCOPES,
        cache_path=_CACHE,
        open_browser=True,
    )
    return spotipy.Spotify(auth_manager=auth)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_now_playing() -> str:
    """Return the currently playing track and playback state."""
    sp = _client()
    current = sp.current_playback()

    if not current or not current.get("is_playing") and not current.get("item"):
        return "Nothing is currently playing on Spotify."

    item    = current.get("item") or {}
    artists = ", ".join(a["name"] for a in item.get("artists", []))
    track   = item.get("name", "Unknown")
    album   = item.get("album", {}).get("name", "")
    playing = current.get("is_playing", False)
    device  = current.get("device", {})
    dev_name = device.get("name", "unknown device")
    dev_vol  = device.get("volume_percent", "?")

    progress_ms = current.get("progress_ms", 0)
    duration_ms = item.get("duration_ms", 1)
    progress_s  = progress_ms // 1000
    duration_s  = duration_ms // 1000
    pos = f"{progress_s // 60}:{progress_s % 60:02d} / {duration_s // 60}:{duration_s % 60:02d}"

    status = "Playing" if playing else "Paused"
    lines = [
        f"{status}: {track} — {artists}",
        f"  Album: {album}",
        f"  Position: {pos}",
        f"  Device: {dev_name}  (volume {dev_vol}%)",
    ]
    return "\n".join(lines)


def get_devices() -> str:
    """List all active Spotify Connect devices."""
    sp = _client()
    result = sp.devices()
    devices = result.get("devices", [])

    if not devices:
        return "No active Spotify devices found. Open Spotify on any device first."

    lines = []
    for d in devices:
        active = " [active]" if d.get("is_active") else ""
        lines.append(
            f"- {d['name']}  ({d['type']})  vol {d.get('volume_percent', '?')}%{active}  id:{d['id']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

def control_playback(action: str, device_id: str = "") -> str:
    """
    Control Spotify playback.

    action: one of play, pause, next, previous
    device_id: optional — target a specific device id from get_devices().
               Defaults to the currently active device.
    """
    sp  = _client()
    did = device_id or None

    action = action.lower().strip()
    if action == "play":
        sp.start_playback(device_id=did)
        return "Playback started."
    elif action == "pause":
        sp.pause_playback(device_id=did)
        return "Playback paused."
    elif action == "next":
        sp.next_track(device_id=did)
        return "Skipped to next track."
    elif action == "previous":
        sp.previous_track(device_id=did)
        return "Went back to previous track."
    else:
        return f"Unknown action '{action}'. Use: play, pause, next, previous."


def set_volume(volume_percent: int, device_id: str = "") -> str:
    """
    Set Spotify volume (0–100).

    device_id: optional — target a specific device.
    """
    volume_percent = max(0, min(100, volume_percent))
    sp  = _client()
    did = device_id or None
    sp.volume(volume_percent, device_id=did)
    return f"Volume set to {volume_percent}%."


def search_and_play(query: str, search_type: str = "track", device_id: str = "") -> str:
    """
    Search Spotify and immediately play the top result.

    query:       e.g. 'Bohemian Rhapsody', 'The Beatles', 'Chill vibes playlist'
    search_type: track | artist | album | playlist  (default: track)
    device_id:   optional — target a specific device id.
    """
    sp    = _client()
    did   = device_id or None
    stype = search_type.lower().strip()
    if stype not in {"track", "artist", "album", "playlist"}:
        stype = "track"

    results = sp.search(q=query, type=stype, limit=1)
    items   = results.get(f"{stype}s", {}).get("items", [])

    if not items:
        return f"No {stype} found for '{query}'."

    item = items[0]
    uri  = item["uri"]
    name = item["name"]

    if stype == "track":
        sp.start_playback(device_id=did, uris=[uri])
        artists = ", ".join(a["name"] for a in item.get("artists", []))
        return f"Now playing: {name} — {artists}"
    else:
        sp.start_playback(device_id=did, context_uri=uri)
        return f"Now playing {stype}: {name}"
