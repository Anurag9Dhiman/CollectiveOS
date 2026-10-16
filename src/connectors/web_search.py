"""
web_search.py
-------------
Web search connector.

Primary:  Brave Search API  (set BRAVE_API_KEY in env)
Fallback: DuckDuckGo Instant Answer API  (no key required, limited results)

Returns a plain-text summary of the top results so the agent can answer
questions about current events, look up documentation, or research topics
without navigating a browser for every query.
"""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
import json


_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
_DDG_URL   = "https://api.duckduckgo.com/"
_TIMEOUT   = 10  # seconds


def web_search(query: str, num_results: int = 5) -> str:
    """Search the web and return a summary of the top results."""
    query = query.strip()
    if not query:
        return "[ERROR: query is empty]"

    num_results = max(1, min(num_results, 10))

    brave_key = os.environ.get("BRAVE_API_KEY", "").strip()
    if brave_key:
        return _brave_search(query, num_results, brave_key)
    return _ddg_search(query)


def _brave_search(query: str, num_results: int, api_key: str) -> str:
    params = urllib.parse.urlencode({
        "q": query,
        "count": num_results,
        "safesearch": "moderate",
        "text_decorations": "false",
    })
    req = urllib.request.Request(
        f"{_BRAVE_URL}?{params}",
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read()
            # urllib doesn't auto-decompress; handle gzip if present
            if resp.getheader("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            data = json.loads(raw)
    except Exception as exc:
        return f"[search error] {exc}"

    results = data.get("web", {}).get("results", [])
    if not results:
        return f"No results found for: {query}"

    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results[:num_results], 1):
        title = r.get("title", "")
        url   = r.get("url", "")
        desc  = r.get("description", "").replace("\n", " ")
        lines.append(f"{i}. {title}\n   {url}\n   {desc}\n")

    return "\n".join(lines)


def _ddg_search(query: str) -> str:
    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
    })
    req = urllib.request.Request(
        f"{_DDG_URL}?{params}",
        headers={"User-Agent": "CollectiveOS/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        return f"[search error] {exc}"

    lines = [f"Search results for: {query}\n"]
    added = 0

    abstract = data.get("AbstractText", "").strip()
    if abstract:
        lines.append(f"Summary: {abstract}")
        src = data.get("AbstractURL", "")
        if src:
            lines.append(f"Source: {src}")
        lines.append("")
        added += 1

    for r in data.get("RelatedTopics", []):
        if added >= 5:
            break
        text = r.get("Text", "").strip()
        url  = r.get("FirstURL", "")
        if text:
            lines.append(f"• {text}")
            if url:
                lines.append(f"  {url}")
            added += 1

    if added == 0:
        return (
            f"No instant-answer results for: {query}. "
            "Set BRAVE_API_KEY for full web search results."
        )

    return "\n".join(lines)
