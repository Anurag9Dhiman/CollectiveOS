"""
Web search connector — real-time information via Tavily.

Tavily is purpose-built for AI agents: it returns pre-parsed, relevant
snippets rather than raw HTML, keeping token usage low and results clean.

Setup
-----
1. Sign up at https://tavily.com (free tier: 1 000 searches / month).
2. Add TAVILY_API_KEY=tvly-... to your .env file.
"""

import os
from functools import lru_cache

from tavily import TavilyClient


@lru_cache(maxsize=1)
def _client() -> TavilyClient:
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        raise RuntimeError(
            "TAVILY_API_KEY not set. "
            "Sign up at https://tavily.com and add the key to .env."
        )
    return TavilyClient(api_key=key)


def search(query: str, max_results: int = 5) -> str:
    """
    Search the web and return a synthesised answer with supporting sources.

    Tavily returns a direct AI-generated answer drawn from the top results,
    plus individual source summaries — ideal for LLM consumption.
    """
    try:
        client = _client()
        resp = client.search(
            query=query,
            max_results=max_results,
            include_answer=True,
        )
    except RuntimeError as e:
        return str(e)
    except Exception as e:
        return f"Search error: {e}"

    lines = []

    answer = (resp.get("answer") or "").strip()
    if answer:
        lines.append(f"**Answer:** {answer}")

    results = resp.get("results", [])
    if results:
        lines.append("\n**Sources:**")
        for i, r in enumerate(results, 1):
            title   = (r.get("title") or "").strip()
            url     = (r.get("url") or "").strip()
            content = (r.get("content") or "").strip()[:300]
            if content and not content.endswith((".", "!", "?")):
                content += "…"
            lines.append(f"{i}. **{title}**\n   {url}\n   {content}")

    return "\n\n".join(lines) if lines else "No results found."
