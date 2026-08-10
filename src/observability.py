"""
Structured observability — token usage tracking, tool latency logging.

Two Postgres tables:
  api_usage   — every Anthropic API call with token counts + estimated cost
  tool_calls  — every tool execution with latency and success/failure

Call configure_logging() once at startup (done in api.py lifespan).
Then use log_api_call() and log_tool_call() anywhere in the stack.
usage_summary() is exposed as an agent tool so the assistant can report costs.
"""

import logging
import os
import time

import structlog

from src.db import connect

# ---------------------------------------------------------------------------
# Pricing (USD per million tokens)
# ---------------------------------------------------------------------------

_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6":         {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.25,  "output": 1.25},
    "claude-opus-4-8":           {"input": 15.00, "output": 75.00},
}
_DEFAULT_PRICE = {"input": 3.00, "output": 15.00}

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_USAGE = """
CREATE TABLE IF NOT EXISTS api_usage (
    id            SERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model         TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'chat',
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd      NUMERIC(10, 6) NOT NULL DEFAULT 0
);
"""

_CREATE_TOOL_CALLS = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id         SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tool_name  TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    success    BOOLEAN NOT NULL DEFAULT TRUE,
    error_type TEXT
);
"""


def _bootstrap() -> None:
    try:
        conn = connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE_USAGE)
                cur.execute(_CREATE_TOOL_CALLS)
        conn.close()
    except Exception:
        pass


_bootstrap()

# ---------------------------------------------------------------------------
# structlog configuration
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    """Configure structlog. Call once at process startup."""
    json_logs = os.environ.get("LOG_FORMAT", "").lower() == "json"

    shared = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer = structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared + [renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(format="%(message)s", level=logging.INFO)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

log = structlog.get_logger()


def log_api_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    source: str = "chat",
) -> None:
    """Record one Anthropic API call — tokens used + estimated cost."""
    p = _PRICING.get(model, _DEFAULT_PRICE)
    cost = (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
    try:
        conn = connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO api_usage (model, source, input_tokens, output_tokens, cost_usd) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (model, source, input_tokens, output_tokens, round(cost, 6)),
                )
        conn.close()
    except Exception:
        pass
    log.info(
        "api_call",
        model=model,
        source=source,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 6),
    )


def log_tool_call(
    tool_name: str,
    latency_ms: int,
    success: bool,
    error_type: str | None = None,
) -> None:
    """Record one tool execution — latency and outcome."""
    try:
        conn = connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO tool_calls (tool_name, latency_ms, success, error_type) "
                    "VALUES (%s, %s, %s, %s)",
                    (tool_name, latency_ms, success, error_type),
                )
        conn.close()
    except Exception:
        pass
    log.info(
        "tool_call",
        tool=tool_name,
        latency_ms=latency_ms,
        success=success,
        error_type=error_type,
    )


def usage_summary(days: int = 1) -> str:
    """
    Return a formatted cost and latency report for the last N days.
    Exposed as an agent tool so the assistant can answer 'how much did I spend today?'
    """
    try:
        conn = connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COALESCE(SUM(input_tokens),  0)::bigint AS total_input,
                    COALESCE(SUM(output_tokens), 0)::bigint AS total_output,
                    COALESCE(SUM(cost_usd),      0)         AS total_cost,
                    COUNT(*)::int                            AS total_calls,
                    model
                FROM api_usage
                WHERE created_at >= NOW() - (%s || ' days')::interval
                GROUP BY model
                ORDER BY SUM(cost_usd) DESC
                """,
                (str(days),),
            )
            model_rows = cur.fetchall()

            cur.execute(
                """
                SELECT
                    tool_name,
                    ROUND(AVG(latency_ms))::int                                          AS avg_ms,
                    COUNT(*)::int                                                         AS calls,
                    ROUND(100.0 * SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*))::int AS success_pct
                FROM tool_calls
                WHERE created_at >= NOW() - (%s || ' days')::interval
                GROUP BY tool_name
                ORDER BY calls DESC
                LIMIT 10
                """,
                (str(days),),
            )
            tool_rows = cur.fetchall()
        conn.close()
    except Exception as exc:
        return f"[ERROR: observability — {exc}]"

    lines = [f"**Usage summary — last {days} day(s)**", ""]

    if not model_rows:
        lines.append("No API calls recorded yet.")
    else:
        total_cost = sum(float(r[2]) for r in model_rows)
        lines.append(f"Total estimated cost: **${total_cost:.4f}**")
        lines.append("")
        lines.append("By model:")
        for total_input, total_output, cost, calls, model in model_rows:
            lines.append(
                f"  • {model}: {calls} calls · "
                f"{total_input:,} in + {total_output:,} out tokens · "
                f"${float(cost):.4f}"
            )

    if tool_rows:
        lines += ["", "Top tools (by call count):"]
        for tool_name, avg_ms, calls, success_pct in tool_rows:
            lines.append(
                f"  • {tool_name}: {calls} calls · {avg_ms}ms avg · {success_pct}% success"
            )

    return "\n".join(lines)
