"""iOS push notification connector via APNs HTTP/2 provider API.

Required env vars:
  APNS_KEY_ID        — 10-char key ID (Apple Developer → Certificates → Keys)
  APNS_TEAM_ID       — 10-char team ID (Apple Developer → Membership)
  APNS_AUTH_KEY_PATH — local path to the .p8 private key file from Apple
  APNS_BUNDLE_ID     — iOS app bundle ID, e.g. com.yourname.collectiveos
  APNS_DEVICE_TOKEN  — 64-char hex token from didRegisterForRemoteNotificationsWithDeviceToken

Optional:
  APNS_SANDBOX=1 — use the development APNs endpoint (sandbox apps)

The JWT provider token is cached for 55 minutes; Apple's limit is 60.
"""

from __future__ import annotations

import os
import time

_jwt_cache: dict[str, object] = {}


def _get_jwt() -> str:
    now = time.time()
    if _jwt_cache.get("token") and now < float(_jwt_cache.get("exp", 0)):
        return str(_jwt_cache["token"])

    import jwt  # PyJWT

    key_id   = os.environ["APNS_KEY_ID"]
    team_id  = os.environ["APNS_TEAM_ID"]
    key_path = os.environ["APNS_AUTH_KEY_PATH"]

    with open(key_path) as fh:
        private_key = fh.read()

    token = jwt.encode(
        {"iss": team_id, "iat": int(now)},
        private_key,
        algorithm="ES256",
        headers={"kid": key_id},
    )
    _jwt_cache["token"] = token
    _jwt_cache["exp"] = now + 55 * 60
    return token


def push_notification(
    title: str,
    body: str,
    badge: int | None = None,
    sound: str = "default",
) -> str:
    """Send a push notification to the user's iPhone."""
    import httpx

    device_token = os.environ.get("APNS_DEVICE_TOKEN", "").strip()
    bundle_id    = os.environ.get("APNS_BUNDLE_ID", "").strip()
    sandbox      = os.environ.get("APNS_SANDBOX", "").lower() in ("1", "true", "yes")

    if not device_token:
        return "[ERROR: APNS_DEVICE_TOKEN is not set]"
    if not bundle_id:
        return "[ERROR: APNS_BUNDLE_ID is not set]"

    try:
        jwt_token = _get_jwt()
    except KeyError as exc:
        return f"[ERROR: missing env var {exc} — check APNS_KEY_ID, APNS_TEAM_ID, APNS_AUTH_KEY_PATH]"
    except FileNotFoundError as exc:
        return f"[ERROR: .p8 key file not found — {exc}]"
    except Exception as exc:
        return f"[ERROR: could not build APNs JWT — {exc}]"

    host = "api.development.push.apple.com" if sandbox else "api.push.apple.com"
    url  = f"https://{host}/3/device/{device_token}"

    aps: dict = {"alert": {"title": title, "body": body}, "sound": sound}
    if badge is not None:
        aps["badge"] = badge

    try:
        with httpx.Client(http2=True, timeout=10.0) as client:
            resp = client.post(
                url,
                headers={
                    "Authorization": f"bearer {jwt_token}",
                    "apns-topic": bundle_id,
                    "apns-push-type": "alert",
                    "apns-priority": "10",
                },
                json={"aps": aps},
            )

        if resp.status_code == 200:
            return f'Push sent: "{title}" — {body}'

        reason = resp.text
        try:
            reason = resp.json().get("reason", reason)
        except Exception:
            pass
        return f"[ERROR: APNs rejected push (HTTP {resp.status_code}) — {reason}]"

    except Exception as exc:
        return f"[ERROR: push failed — {exc}]"
