#!/usr/bin/env python3
"""
RQ18: Connector Scalability Evaluation

Tests how tool-selection accuracy degrades as the connector library grows.
Uses a synthetic connector registry with 5 → 10 → 25 → 50 → 100 tools,
then asks Gemini to select the right tool for each query.

Metrics:
  tool_selection_accuracy  = correct selections / total queries
  wrong_tool_rate          = incorrect selections / total queries
  avg_tokens_consumed      = token cost per selection decision
  avg_latency_s            = selection latency

Usage:
    python3 scripts/eval_connector_scale.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

for _line in (_ROOT / ".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        k, _, v = _line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


# ── Connector registry (100 synthetic tools) ─────────────────────────────────

BASE_CONNECTORS = [
    # These 25 are the "real" ones that map to existing code
    ("mac_system",       "Control macOS system settings: volume, brightness, wifi, bluetooth, DND, dark mode"),
    ("local_files",      "Read, write, list, search files on the local filesystem"),
    ("web_search",       "Search the web using DuckDuckGo or Brave and return results"),
    ("calendar",         "Read and create events in Apple Calendar or Google Calendar"),
    ("email",            "Read, send, and manage email via Gmail or Apple Mail"),
    ("notes",            "Create, read, search notes in Apple Notes or Notion"),
    ("reminders",        "Create and manage reminders and to-do items"),
    ("contacts",         "Look up and manage contacts in Apple Contacts"),
    ("messages",         "Send and read iMessage or SMS messages"),
    ("spotify",          "Control Spotify: play, pause, skip, search, playlist management"),
    ("home_assistant",   "Control smart home devices via Home Assistant"),
    ("health",           "Read Apple Health data: steps, heart rate, sleep, workouts"),
    ("weather",          "Get current weather and forecasts for any location"),
    ("maps",             "Get directions, estimated travel time, nearby places"),
    ("browser_nav",      "Open URLs, navigate, extract text from web pages"),
    ("clipboard",        "Read and write the system clipboard"),
    ("screenshot",       "Take screenshots, annotate, share screen content"),
    ("terminal",         "Execute shell commands in Terminal"),
    ("code_editor",      "Open files in VS Code, search code, run tasks"),
    ("github",           "Search repos, issues, PRs; get file content from GitHub"),
    ("slack",            "Send messages, read channels, search Slack workspace"),
    ("notion",           "Read and edit Notion pages, databases, and tasks"),
    ("google_drive",     "Read, list, upload, download Google Drive files"),
    ("youtube",          "Search YouTube, get video details and transcripts"),
    ("wolfram",          "Compute mathematical expressions and get factual answers"),
]

# Additional synthetic connectors to pad to 100
EXTRA_CONNECTORS = [
    ("twitter",          "Post tweets, search timeline, follow/unfollow users"),
    ("linkedin",         "Search professionals, read posts, send connection requests"),
    ("reddit",           "Search subreddits, read posts, comment on Reddit"),
    ("instagram",        "View and like posts on Instagram"),
    ("facebook",         "Read Facebook feed and events"),
    ("amazon",           "Search Amazon products, add to cart, check orders"),
    ("ebay",             "Search eBay listings and track auctions"),
    ("stripe",           "Check Stripe balance, refunds, payment history"),
    ("paypal",           "View PayPal transactions and send money"),
    ("coinbase",         "Check crypto portfolio and market prices"),
    ("robinhood",        "View stock portfolio and execute trades"),
    ("airbnb",           "Search vacation rentals and manage bookings"),
    ("uber",             "Request rides and track driver location"),
    ("doordash",         "Order food delivery and track order status"),
    ("yelp",             "Search restaurants and read reviews"),
    ("opentable",        "Make restaurant reservations"),
    ("ticketmaster",     "Search and purchase event tickets"),
    ("netflix",          "Search Netflix catalog and get recommendations"),
    ("hulu",             "Browse Hulu shows and movies"),
    ("disney_plus",      "Search Disney+ content and watchlist"),
    ("apple_tv",         "Browse Apple TV+ catalog and play content"),
    ("audible",          "Search audiobooks and manage library"),
    ("kindle",           "Access Kindle library and read ebooks"),
    ("goodreads",        "Track books, get reading recommendations"),
    ("medium",           "Read and search Medium articles"),
    ("substack",         "Read subscribed Substack newsletters"),
    ("pocket",           "Save and retrieve web articles for later reading"),
    ("feedly",           "Read RSS feeds and follow publications"),
    ("evernote",         "Search and create Evernote notebooks"),
    ("bear",             "Create and search Bear markdown notes"),
    ("obsidian",         "Access Obsidian vault notes and links"),
    ("todoist",          "Manage tasks in Todoist"),
    ("things3",          "Manage tasks in Things 3"),
    ("omnifocus",        "Manage projects in OmniFocus"),
    ("linear",           "Create and track issues in Linear"),
    ("jira",             "Manage Jira tickets and sprints"),
    ("asana",            "Create and assign tasks in Asana"),
    ("trello",           "Manage Trello boards and cards"),
    ("monday",           "Manage Monday.com boards and tasks"),
    ("airtable",         "Read and write Airtable bases"),
    ("figma",            "Access Figma design files and comments"),
    ("miro",             "Read and edit Miro whiteboards"),
    ("zoom",             "Schedule and join Zoom meetings"),
    ("google_meet",      "Start and join Google Meet calls"),
    ("calendly",         "Schedule appointments via Calendly"),
    ("typeform",         "Create and read Typeform survey responses"),
    ("google_forms",     "Create and read Google Forms submissions"),
    ("surveymonkey",     "Create and analyze SurveyMonkey surveys"),
    ("mailchimp",        "Manage email lists and campaigns in Mailchimp"),
    ("sendgrid",         "Send transactional emails via SendGrid API"),
    ("twilio",           "Send SMS and make voice calls via Twilio"),
    ("openai",           "Call OpenAI GPT models for text generation"),
    ("anthropic",        "Call Claude models via Anthropic API"),
    ("google_ai",        "Call Gemini models via Google AI API"),
    ("huggingface",      "Run open-source models via HuggingFace API"),
    ("replicate",        "Run AI models on Replicate cloud GPU"),
    ("aws_s3",           "List, upload, download files from AWS S3"),
    ("google_cloud",     "Manage Google Cloud resources and GCS files"),
    ("azure_blob",       "Access Azure Blob Storage containers"),
    ("cloudflare",       "Manage Cloudflare DNS records and pages"),
    ("vercel",           "Deploy and manage Vercel projects"),
    ("netlify",          "Deploy and manage Netlify sites"),
    ("heroku",           "Manage Heroku apps and dynos"),
    ("docker",           "List, start, stop Docker containers"),
    ("kubernetes",       "Manage Kubernetes pods and deployments"),
    ("postgres_db",      "Query and update a local PostgreSQL database"),
    ("mysql_db",         "Query and update a MySQL database"),
    ("mongodb",          "Query MongoDB collections"),
    ("redis_store",      "Get and set Redis key-value pairs"),
    ("elasticsearch",    "Search and index documents in Elasticsearch"),
    ("prometheus",       "Query Prometheus metrics and create alerts"),
    ("grafana",          "Read Grafana dashboards and panels"),
    ("datadog",          "View Datadog APM traces and logs"),
    ("sentry",           "Search Sentry error events and releases"),
    ("pagerduty",        "Manage on-call schedules and incidents in PagerDuty"),
]

ALL_CONNECTORS = BASE_CONNECTORS + EXTRA_CONNECTORS  # 100 total

# ── Test queries with correct tool ────────────────────────────────────────────
TEST_QUERIES = [
    ("Turn down the system volume to 30%",              "mac_system"),
    ("Search the web for Python async tutorial",        "web_search"),
    ("Add a meeting to my calendar for tomorrow at 3pm","calendar"),
    ("Send an email to alice@example.com",              "email"),
    ("Create a new note about the project",             "notes"),
    ("Play some music on Spotify",                      "spotify"),
    ("Turn off the living room lights",                 "home_assistant"),
    ("How many steps have I taken today?",              "health"),
    ("What is the weather in San Francisco?",           "weather"),
    ("Get directions to the nearest coffee shop",       "maps"),
    ("Open amazon.com in the browser",                  "browser_nav"),
    ("Run git status in the terminal",                  "terminal"),
    ("Search GitHub for langgraph issues",              "github"),
    ("Send a message to the team channel on Slack",     "slack"),
    ("Copy this text to clipboard",                     "clipboard"),
]


def _ask_gemini_select_tool(query: str, available_tools: list[tuple[str, str]]) -> tuple[str, int]:
    """Ask Gemini to select the best tool. Returns (selected_tool_id, tokens_used)."""
    from google import genai
    from google.genai import types as gtypes

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""))
    tools_text = "\n".join(f"  {i+1:3}. {tid:<25} — {desc}" for i, (tid, desc) in enumerate(available_tools))
    prompt = (
        f"User request: {query}\n\n"
        f"Available tools:\n{tools_text}\n\n"
        f"Which single tool id best handles this request? "
        f"Reply with ONLY the tool id, nothing else."
    )
    resp = client.models.generate_content(
        model="models/gemini-3.6-flash",
        contents=prompt,
    )
    selected = resp.text.strip().strip('"').strip("'").lower()
    tokens = (resp.usage_metadata.prompt_token_count or 0) + (resp.usage_metadata.candidates_token_count or 0)
    # Normalize: sometimes the model returns the tool name with extra words
    for tid, _ in available_tools:
        if tid.lower() in selected:
            return tid, tokens
    return selected, tokens


def run_rq18() -> None:
    print("\n=== RQ18: Connector Scalability ===\n")

    library_sizes = [5, 10, 25, 50, 100]
    size_results: dict[int, dict] = {}

    for size in library_sizes:
        # Take the first `size` connectors (always includes the 25 "real" ones at front)
        available = ALL_CONNECTORS[:size]
        # Only test queries for which the correct tool is in this subset
        valid_queries = [(q, t) for q, t in TEST_QUERIES
                         if any(cid == t for cid, _ in available)]

        if not valid_queries:
            print(f"  {size:3} tools — no valid queries, skipping")
            continue

        correct, total = 0, 0
        tokens_list, lat_list = [], []
        errors = []

        for query, expected in valid_queries:
            t0 = time.time()
            try:
                selected, tokens = _ask_gemini_select_tool(query, available)
                lat = round(time.time() - t0, 2)
                is_correct = selected.lower() == expected.lower()
                correct += int(is_correct)
                total += 1
                tokens_list.append(tokens)
                lat_list.append(lat)
                if not is_correct:
                    errors.append({"query": query[:50], "expected": expected, "got": selected})
            except Exception as exc:
                total += 1
                errors.append({"query": query[:50], "expected": expected, "error": str(exc)[:80]})
            time.sleep(5)  # Gemini rate limit

        acc = round(correct / total, 3) if total else 0
        avg_tok = round(sum(tokens_list) / max(1, len(tokens_list)))
        avg_lat = round(sum(lat_list) / max(1, len(lat_list)), 2)
        size_results[size] = {
            "accuracy": acc, "correct": correct, "total": total,
            "avg_tokens": avg_tok, "avg_latency_s": avg_lat, "errors": errors,
        }
        print(f"  {size:3} tools — accuracy={acc:.0%}  ({correct}/{total})  "
              f"tokens={avg_tok}  lat={avg_lat:.1f}s")

    print(f"\n  Scalability curve:")
    print(f"  {'Tools':>6}  {'Accuracy':>9}  {'Tokens':>8}  {'Latency':>9}")
    print("  " + "─" * 40)
    for sz in library_sizes:
        if sz in size_results:
            m = size_results[sz]
            print(f"  {sz:>6}  {m['accuracy']:>9.0%}  {m['avg_tokens']:>8}  {m['avg_latency_s']:>9.2f}s")

    out = {"rq": "RQ18", "library_sizes": size_results}
    Path("eval_results_rq18_connector_scale.json").write_text(json.dumps(out, indent=2))
    print("\nResults saved → eval_results_rq18_connector_scale.json")


if __name__ == "__main__":
    run_rq18()
