"""
Finance connector (read-only) — account balances, transactions, and spending
summaries via Plaid.

Env vars:
  PLAID_CLIENT_ID    — from https://dashboard.plaid.com/
  PLAID_SECRET       — environment-specific secret (sandbox / development / production)
  PLAID_ACCESS_TOKEN — stored after completing the Plaid Link flow (see setup below)
  PLAID_ENV          — 'sandbox', 'development', or 'production' (default: development)

One-time setup (Plaid Link flow):
  1. Install Plaid Quickstart: https://github.com/plaid/quickstart
     OR run the minimal setup: python src/connectors/finance_setup.py
  2. Connect your bank through the Link UI.
  3. Copy the printed access_token to PLAID_ACCESS_TOKEN in .env.
  Sandbox testing: use credentials user_good / pass_good at any bank.
"""

import datetime
import os
from collections import defaultdict

import requests

_ENVS = {
    "sandbox":     "https://sandbox.plaid.com",
    "development": "https://development.plaid.com",
    "production":  "https://production.plaid.com",
}


def _base() -> str:
    return _ENVS.get(os.environ.get("PLAID_ENV", "development").lower(), _ENVS["development"])


def _check_config() -> str | None:
    missing = [k for k in ("PLAID_CLIENT_ID", "PLAID_SECRET", "PLAID_ACCESS_TOKEN")
               if not os.environ.get(k)]
    if missing:
        return (
            f"Plaid not configured — missing: {', '.join(missing)}. "
            "See .env.example for setup instructions."
        )
    return None


def _post(endpoint: str, extra: dict | None = None) -> dict:
    body = {
        "client_id":    os.environ.get("PLAID_CLIENT_ID", ""),
        "secret":       os.environ.get("PLAID_SECRET", ""),
        "access_token": os.environ.get("PLAID_ACCESS_TOKEN", ""),
    }
    if extra:
        body.update(extra)
    resp = requests.post(f"{_base()}{endpoint}", json=body, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if "error_code" in data:
        raise RuntimeError(
            f"Plaid {data['error_code']}: {data.get('error_message', '(no message)')}"
        )
    return data


def _fmt(amount: float) -> str:
    """Plaid convention: positive = debit (money out), negative = credit (money in)."""
    if amount >= 0:
        return f"${amount:,.2f}"
    return f"+${abs(amount):,.2f}"  # refund / income


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def finance_get_accounts() -> str:
    """List connected bank/credit accounts with real-time balances."""
    err = _check_config()
    if err:
        return err
    try:
        data = _post("/accounts/balance/get")
        accounts = data.get("accounts", [])
        if not accounts:
            return "No accounts found."
        lines = ["Connected accounts:"]
        for a in accounts:
            name = a.get("official_name") or a.get("name", "Unknown")
            mask = a.get("mask", "????")
            atype = f"{a.get('type', '').title()} / {a.get('subtype', '').replace('_', ' ').title()}"
            bal = a.get("balances", {})
            current   = bal.get("current")
            available = bal.get("available")
            curr_str  = f"${current:,.2f}"   if current   is not None else "—"
            avail_str = f"${available:,.2f}" if available is not None else "—"
            lines.append(
                f"  {name} (••••{mask})  [{atype}]\n"
                f"    current: {curr_str}   available: {avail_str}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Finance accounts error: {exc}"


def finance_get_transactions(days: int = 30, account_id: str = "") -> str:
    """Fetch recent transactions, newest first. Optionally filter to one account ID."""
    err = _check_config()
    if err:
        return err
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=days)).isoformat()
    end   = today.isoformat()
    extra: dict = {"start_date": start, "end_date": end, "count": 250}
    if account_id:
        extra["options"] = {"account_ids": [account_id.strip()]}
    try:
        data = _post("/transactions/get", extra)
        txns = data.get("transactions", [])
        if not txns:
            return f"No transactions found in the last {days} days."
        lines = [f"Transactions — last {days} days ({len(txns)} shown):"]
        for t in txns:
            pending = " (pending)" if t.get("pending") else ""
            cat     = t.get("category") or []
            cat_str = f"  [{' > '.join(cat)}]" if cat else ""
            merchant = t.get("merchant_name") or t.get("name", "Unknown")
            lines.append(
                f"  {t['date']}  {_fmt(t.get('amount', 0)):>12}  "
                f"{merchant}{pending}{cat_str}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Finance transactions error: {exc}"


def finance_get_spending_summary(days: int = 30) -> str:
    """Summarize spending by top-level Plaid category over the last N days."""
    err = _check_config()
    if err:
        return err
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=days)).isoformat()
    end   = today.isoformat()
    try:
        data = _post("/transactions/get", {"start_date": start, "end_date": end, "count": 500})
        txns = data.get("transactions", [])
        spending: dict[str, float] = defaultdict(float)
        total = 0.0
        for t in txns:
            if t.get("pending"):
                continue
            amount = t.get("amount", 0.0)
            if amount <= 0:
                continue  # skip credits / refunds
            cat     = t.get("category") or ["Uncategorized"]
            top_cat = cat[0]
            spending[top_cat] += amount
            total += amount
        if not spending:
            return f"No spending found in the last {days} days."
        lines = [f"Spending summary — last {days} days  (total: ${total:,.2f}):"]
        for cat, amt in sorted(spending.items(), key=lambda x: -x[1]):
            pct = amt / total * 100
            lines.append(f"  {cat:<32} ${amt:>10,.2f}  ({pct:.1f}%)")
        return "\n".join(lines)
    except Exception as exc:
        return f"Finance spending summary error: {exc}"
