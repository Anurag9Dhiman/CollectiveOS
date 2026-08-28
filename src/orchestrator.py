"""
Task orchestrator — plan and execute multi-step agentic tasks.

Flow:
  plan_and_run(description)
    1. LLM call → ordered list of (tool, args, reason) steps
    2. Steps written to task_steps table (pending)
    3. Each step executed via _exec_tool_fn(), DB status updated as it runs
    4. Summary returned to the agent

State machine (matches schema.sql CHECK constraint on tasks.status):
  pending → planning → running → completed | failed | cancelled

This is a single-LLM planner, not a multi-agent system. One Gemini call
does all the planning; the existing connector functions handle execution.
"""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src import db as _db

PLANNER_MODEL = os.environ.get(
    "GEMINI_PLANNER_MODEL",
    os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
)

MAX_STEPS = 10  # hard cap — prevents runaway planning


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class _Step(BaseModel):
    tool: str = Field(description="Exact tool function name")
    args: dict[str, Any] = Field(default_factory=dict, description="Keyword arguments for the tool")
    reason: str = Field(default="", description="One sentence: why this step is needed")


class _Plan(BaseModel):
    steps: list[_Step] = Field(description="Ordered list of tool calls to accomplish the task")


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = (
    "You are a task planner for a personal AI assistant. "
    "Given a task description and the available tools, produce an ordered list "
    "of tool calls that accomplish the task end to end. "
    "For each step specify: tool (exact function name), args (JSON object of "
    "keyword arguments matching the tool's signature), and reason (one sentence). "
    "Use only tools from the provided list. Maximum {max_steps} steps. "
    "If the task cannot be done with the available tools, return an empty steps list."
)


def _plan(description: str, available_tools: list[dict]) -> list[_Step]:
    tool_lines = "\n".join(
        f"- {t['name']}: {t.get('description', '').split('.')[0]}"
        for t in available_tools
    )
    llm = ChatGoogleGenerativeAI(
        model=PLANNER_MODEL,
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
    )
    result: _Plan = llm.with_structured_output(_Plan).invoke([
        SystemMessage(content=_PLAN_SYSTEM.format(max_steps=MAX_STEPS)),
        HumanMessage(content=f"Task: {description}\n\nAvailable tools:\n{tool_lines}"),
    ])
    return result.steps[:MAX_STEPS]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _create_task(description: str) -> int:
    conn = _db.connect()
    try:
        uid = _db.default_user_id(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tasks (user_id, description, status) "
                "VALUES (%s, %s, 'pending') RETURNING id",
                (uid, description),
            )
            task_id: int = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return task_id


def _set_task_status(task_id: int, status: str) -> None:
    conn = _db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE tasks SET status = %s WHERE id = %s", (status, task_id))
        conn.commit()
    finally:
        conn.close()


def _insert_steps(task_id: int, steps: list[_Step]) -> list[int]:
    conn = _db.connect()
    try:
        ids: list[int] = []
        with conn.cursor() as cur:
            for step in steps:
                cur.execute(
                    "INSERT INTO task_steps (task_id, tool_name, input, status) "
                    "VALUES (%s, %s, %s, 'pending') RETURNING id",
                    (task_id, step.tool, json.dumps(step.args)),
                )
                ids.append(cur.fetchone()[0])
        conn.commit()
    finally:
        conn.close()
    return ids


def _update_step(step_id: int, status: str, output: str | None = None) -> None:
    conn = _db.connect()
    try:
        with conn.cursor() as cur:
            if output is not None:
                cur.execute(
                    "UPDATE task_steps SET status = %s, output = %s WHERE id = %s",
                    (status, output, step_id),
                )
            else:
                cur.execute(
                    "UPDATE task_steps SET status = %s WHERE id = %s",
                    (status, step_id),
                )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def _execute_steps(task_id: int, steps: list[_Step], step_ids: list[int]) -> tuple[bool, str]:
    from src.assistant_starter import _exec_tool_fn

    lines: list[str] = []
    all_ok = True
    for step, step_id in zip(steps, step_ids):
        _update_step(step_id, "running")
        try:
            output = _exec_tool_fn(step.tool, step.args)
            _update_step(step_id, "completed", str(output))
            lines.append(f"✓ {step.tool}: {str(output)[:300]}")
        except Exception as exc:
            err = f"[ERROR: {exc}]"
            _update_step(step_id, "failed", err)
            lines.append(f"✗ {step.tool}: {err}")
            all_ok = False
    return all_ok, "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_and_run(description: str, available_tools: list[dict] | None = None) -> str:
    """Plan and execute a multi-step task. Returns a text summary."""
    if available_tools is None:
        from src.assistant_starter import TOOLS
        available_tools = TOOLS

    task_id = _create_task(description)
    _set_task_status(task_id, "planning")

    try:
        steps = _plan(description, available_tools)
    except Exception as exc:
        _set_task_status(task_id, "failed")
        return f"[task #{task_id}] Planning failed: {exc}"

    if not steps:
        _set_task_status(task_id, "failed")
        return f"[task #{task_id}] Could not plan steps for: {description}"

    step_ids = _insert_steps(task_id, steps)
    _set_task_status(task_id, "running")

    all_ok, summary = _execute_steps(task_id, steps, step_ids)
    _set_task_status(task_id, "completed" if all_ok else "failed")

    word = "completed" if all_ok else "partially failed"
    return f"Task #{task_id} {word} ({len(steps)} steps):\n{summary}"


def get_task(task_id: int) -> dict:
    conn = _db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, description, status, created_at FROM tasks WHERE id = %s",
                (task_id,),
            )
            row = cur.fetchone()
            if not row:
                return {}
            task = {
                "id": row[0], "description": row[1],
                "status": row[2], "created_at": str(row[3]),
            }
            cur.execute(
                "SELECT id, tool_name, input, output, status "
                "FROM task_steps WHERE task_id = %s ORDER BY id",
                (task_id,),
            )
            task["steps"] = [
                {"id": r[0], "tool": r[1], "args": r[2], "output": r[3], "status": r[4]}
                for r in cur.fetchall()
            ]
    finally:
        conn.close()
    return task


def list_tasks(limit: int = 10) -> list[dict]:
    conn = _db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, description, status, created_at "
                "FROM tasks ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            return [
                {"id": r[0], "description": r[1], "status": r[2], "created_at": str(r[3])}
                for r in cur.fetchall()
            ]
    finally:
        conn.close()


def cancel_task(task_id: int) -> str:
    conn = _db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE tasks SET status = 'cancelled' "
                "WHERE id = %s AND status IN ('pending', 'planning', 'running') "
                "RETURNING id",
                (task_id,),
            )
            updated = cur.fetchone()
        conn.commit()
    finally:
        conn.close()
    if updated:
        return f"Task #{task_id} cancelled."
    return f"Task #{task_id} could not be cancelled (already finished or not found)."
