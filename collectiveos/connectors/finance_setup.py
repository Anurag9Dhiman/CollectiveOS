"""
One-time Plaid Link setup — run this once to get your PLAID_ACCESS_TOKEN.

Usage:
  export PLAID_CLIENT_ID=...
  export PLAID_SECRET=...
  export PLAID_ENV=sandbox   # or development / production
  python src/connectors/finance_setup.py

Opens a local web page at http://localhost:8765 that loads the Plaid Link widget.
After you connect your bank, the access token is printed to the terminal.
Copy it to PLAID_ACCESS_TOKEN in your .env file — that's it.

Sandbox credentials: username user_good, password pass_good (any bank shown).
"""

import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

_CLIENT_ID = os.environ.get("PLAID_CLIENT_ID", "")
_SECRET    = os.environ.get("PLAID_SECRET", "")
_ENV       = os.environ.get("PLAID_ENV", "sandbox").lower()
_BASES     = {
    "sandbox":     "https://sandbox.plaid.com",
    "development": "https://development.plaid.com",
    "production":  "https://production.plaid.com",
}
_BASE = _BASES.get(_ENV, _BASES["sandbox"])
_PORT = 8765


def _create_link_token() -> str:
    resp = requests.post(f"{_BASE}/link/token/create", json={
        "client_id": _CLIENT_ID, "secret": _SECRET,
        "client_name": "CollectiveOS",
        "user": {"client_user_id": "personal"},
        "products": ["transactions"],
        "country_codes": ["US", "CA", "GB"],
        "language": "en",
        "redirect_uri": None,
    }, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if "error_code" in data:
        sys.exit(f"Plaid error: {data['error_code']} — {data.get('error_message')}")
    return data["link_token"]


_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Plaid Link Setup</title>
<style>body{{font-family:system-ui;display:flex;flex-direction:column;
align-items:center;justify-content:center;height:100vh;background:#0f1117;color:#e1e4e8;gap:16px}}
button{{background:#1f6feb;color:#fff;border:none;border-radius:8px;
padding:12px 28px;font-size:1rem;cursor:pointer}}
button:hover{{background:#388bfd}}
#status{{color:#8b949e;font-size:0.9rem}}</style></head>
<body>
<h2>CollectiveOS — Connect Your Bank</h2>
<p id="status">Click the button to open Plaid Link and connect your bank account.</p>
<button onclick="openLink()">Connect Bank</button>
<script src="https://cdn.plaid.com/link/v2/stable/link-initialize.js"></script>
<script>
function openLink() {{
  var handler = Plaid.create({{
    token: '{link_token}',
    onSuccess: function(public_token, metadata) {{
      document.getElementById('status').textContent = 'Exchanging token…';
      fetch('/exchange?public_token=' + encodeURIComponent(public_token))
        .then(r => r.text())
        .then(msg => {{
          document.getElementById('status').innerHTML = msg;
        }});
    }},
    onExit: function(err) {{
      if (err) document.getElementById('status').textContent = 'Error: ' + err.error_message;
    }}
  }});
  handler.open();
}}
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    link_token: str = ""

    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == "/":
            body = _HTML.format(link_token=self.link_token).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/exchange"):
            from urllib.parse import parse_qs, urlparse
            params = parse_qs(urlparse(self.path).query)
            public_token = params.get("public_token", [""])[0]
            resp = requests.post(f"{_BASE}/item/public_token/exchange", json={
                "client_id": _CLIENT_ID, "secret": _SECRET,
                "public_token": public_token,
            }, timeout=15)
            data = resp.json()
            access_token = data.get("access_token", "")
            print(f"\n✅  Access token:\n\n  PLAID_ACCESS_TOKEN={access_token}\n")
            print("Copy the line above into your .env file, then restart the server.\n")
            msg = (
                "✅ Bank connected! Access token printed to terminal.<br>"
                "Copy PLAID_ACCESS_TOKEN=... into your .env and restart the server."
            )
            body = msg.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)


def main():
    if not _CLIENT_ID or not _SECRET:
        sys.exit("Set PLAID_CLIENT_ID and PLAID_SECRET before running this script.")
    print(f"Creating Plaid Link token ({_ENV})…")
    link_token = _create_link_token()
    _Handler.link_token = link_token
    print(f"Opening http://localhost:{_PORT} — connect your bank there.")
    webbrowser.open(f"http://localhost:{_PORT}")
    server = HTTPServer(("localhost", _PORT), _Handler)
    server.handle_request()   # serve the page
    server.handle_request()   # serve the /exchange callback


if __name__ == "__main__":
    main()
