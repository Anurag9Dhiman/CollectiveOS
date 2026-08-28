"""LangGraph agent — CollectiveOS 2026.

Replaces the raw Gemini while-loop with a proper StateGraph:

  START
    └─▶ agent  ──────────────────────────────────────────────────▶ END
               │ (pending read tools)                     ▲
               └─▶ read_tools ──────────────────────────▶ agent
               │ (pending write tools)
               └─▶ write_tools ◀── interrupt_before ──── (user approves)
                                └─▶ agent

Key 2026 properties
───────────────────
• PostgresSaver checkpointer  — full conversation state survives restarts
• interrupt_before=["write_tools"]  — hard HITL gate before any write action
• LangSmith tracing via @traceable on the Gemini call
• Stateless Gemini API (models.generate_content) — history lives in state,
  not in an ephemeral SDK chat object; the graph can pause and resume safely
• Parallel read-tool execution via ThreadPoolExecutor (kept from old loop)

The existing connector functions (TOOL_FUNCTIONS, _exec_tool_fn) are called
unchanged. No connector code needs to know about LangGraph.

Thread ID convention: str(conversation_id) — the same int conversation ID
already stored in the database, stringified for LangGraph's thread config.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal, TypedDict

from google.genai import types as _gtypes
from langsmith import traceable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# Write-tool set — require interrupt_before HITL confirmation
# ---------------------------------------------------------------------------

WRITE_TOOLS: frozenset[str] = frozenset({
    "memory_remember", "memory_forget",
    "create_event", "create_draft", "send_email",
    "add_task", "complete_task", "update_task",
    "control_device", "set_light",
    "spotify_control", "spotify_set_volume", "spotify_search_play",
    "show_notification", "open_application", "set_system_volume",
    "imessage_send", "write_local_file", "browser_open_url",
    "reminders_add", "reminders_complete",
    "notes_create", "notes_append", "clipboard_write", "telegram_send",
    "notion_create_page", "notion_append_to_page",
    "github_create_issue", "slack_send_message",
    "car_lock", "car_climate", "appliances_control",
    "push_notification",
})

MAX_ITER = 10
MAX_HISTORY_ENTRIES = 40   # message entries (user + model combined) before trimming
TRIM_KEEP = 20             # how many recent entries to retain after trimming


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # Gemini-format conversation history: list of {"role": ..., "parts": [...]} dicts.
    # Stored as plain Python dicts so PostgresSaver can JSON-serialise them.
    history: list[dict]
    system_prompt: str
    active_tools: list[dict]  # router-selected subset of TOOLS
    reply: str                 # set when the agent produces a final text answer
    pending_write: list[dict]  # fn-call dicts awaiting HITL approval
    approved: bool | None      # set by /chat/approve; None until asked
    iteration: int             # loop-count guard


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def _history_to_gemini(history: list[dict]) -> list[Any]:
    """Convert serialisable history dicts back to Gemini Content objects.

    Each entry is {"role": "user"|"model", "parts": [...]}.
    Each part is one of:
      {"text": "..."}
      {"function_call": {"name": "...", "args": {...}}}
      {"function_response": {"name": "...", "response": {...}}}
    """
    contents = []
    for msg in history:
        parts = []
        for p in msg.get("parts", []):
            if "text" in p:
                parts.append(_gtypes.Part(text=p["text"]))
            elif "function_call" in p:
                fc = p["function_call"]
                parts.append(_gtypes.Part(
                    function_call=_gtypes.FunctionCall(
                        name=fc["name"],
                        args=fc.get("args", {}),
                    )
                ))
            elif "function_response" in p:
                fr = p["function_response"]
                parts.append(_gtypes.Part(
                    function_response=_gtypes.FunctionResponse(
                        name=fr["name"],
                        response=fr.get("response", {}),
                    )
                ))
        if parts:
            contents.append(_gtypes.Content(role=msg["role"], parts=parts))
    return contents


def _extract_fn_call_dicts(response: Any) -> list[dict]:
    """Pull function calls from a Gemini response as serialisable dicts."""
    calls = []
    try:
        for part in response.candidates[0].content.parts:
            fc = getattr(part, "function_call", None)
            if fc and fc.name:
                calls.append({"name": fc.name, "args": dict(fc.args or {})})
    except (IndexError, AttributeError):
        pass
    return calls


def _response_text(response: Any) -> str:
    try:
        return response.text or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# History trimming — keeps Gemini call cost bounded in long conversations
# ---------------------------------------------------------------------------

def _summarize_history(messages: list[dict]) -> str:
    """Call Gemini to produce a compact summary of older conversation turns."""
    from src.assistant_starter import MODEL, _get_client

    history_objs = _history_to_gemini(messages)
    summarize_prompt = _gtypes.Content(
        role="user",
        parts=[_gtypes.Part(
            text=(
                "Summarize the above conversation compactly. "
                "Keep every key fact, decision, and preference the assistant needs "
                "to continue helping effectively. Omit filler and pleasantries. "
                "Maximum 300 words."
            )
        )],
    )
    try:
        resp = _get_client().models.generate_content(
            model=MODEL,
            contents=history_objs + [summarize_prompt],
        )
        return resp.text or "[summary unavailable]"
    except Exception:
        return "[summary unavailable]"


def _trim_history(history: list[dict]) -> list[dict]:
    """If history exceeds MAX_HISTORY_ENTRIES, summarize the oldest portion.

    The trimmed list is returned (and will be persisted to state), so the
    summarization call happens at most once per trim threshold crossing.
    Returns the original list unchanged when under the threshold.
    """
    if len(history) <= MAX_HISTORY_ENTRIES:
        return history

    # Align cut-point to the first 'user' boundary at or past our target index
    keep_from = len(history) - TRIM_KEEP
    while keep_from < len(history) and history[keep_from]["role"] != "user":
        keep_from += 1

    to_summarize = history[:keep_from]
    recent = history[keep_from:]

    if not to_summarize:
        return history

    summary_text = _summarize_history(to_summarize)
    summary_entry = {
        "role": "user",
        "parts": [{"text": f"[Earlier conversation summary]\n{summary_text}"}],
    }
    return [summary_entry] + recent


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

@traceable(name="gemini_call")
def _call_gemini(
    history: list[Any],
    system_prompt: str,
    active_tools: list[dict],
    model: str,
) -> Any:
    """Single Gemini generate_content call. @traceable → LangSmith traces it."""
    from src.assistant_starter import _to_gemini_tools, _get_client, _obs
    gemini_tools = _to_gemini_tools(active_tools)
    return _get_client().models.generate_content(
        model=model,
        contents=history,
        config=_gtypes.GenerateContentConfig(
            tools=gemini_tools or None,
            system_instruction=system_prompt or None,
        ),
    )


def agent_node(state: AgentState) -> dict:
    """Call Gemini with the current history; classify any tool calls."""
    from src.assistant_starter import MODEL, _obs

    # Trim long histories before the Gemini call; persists the compacted form.
    history = _trim_history(state["history"])

    history_objs = _history_to_gemini(history)
    response = _call_gemini(
        history_objs,
        state["system_prompt"],
        state["active_tools"],
        MODEL,
    )

    try:
        if response.usage_metadata:
            _obs.log_api_call(
                MODEL,
                response.usage_metadata.prompt_token_count,
                response.usage_metadata.candidates_token_count,
            )
    except Exception:
        pass

    fn_calls = _extract_fn_call_dicts(response)
    iteration = state.get("iteration", 0) + 1

    if not fn_calls:
        # Done — no more tool calls
        return {
            "reply": _response_text(response),
            "history": history + [
                {"role": "model", "parts": [{"text": _response_text(response)}]}
            ],
            "iteration": iteration,
            "pending_write": [],
        }

    # Add model's tool-call turn to history
    model_parts = [
        {"function_call": {"name": c["name"], "args": c["args"]}} for c in fn_calls
    ]
    new_history = history + [{"role": "model", "parts": model_parts}]

    write_calls = [c for c in fn_calls if c["name"] in WRITE_TOOLS]
    read_calls  = [c for c in fn_calls if c["name"] not in WRITE_TOOLS]

    # Execute read tools immediately inline; defer write tools for HITL
    if read_calls:
        new_history = _execute_calls(read_calls, new_history)

    if write_calls:
        return {
            "history": new_history,
            "pending_write": write_calls,
            "iteration": iteration,
        }

    # Only read tools — agent loops back automatically
    return {
        "history": new_history,
        "pending_write": [],
        "iteration": iteration,
    }


def _execute_calls(calls: list[dict], history: list[dict]) -> list[dict]:
    """Execute tool calls in parallel and append function-response parts to history."""
    from src.assistant_starter import _exec_tool_fn

    def _run(call: dict) -> dict:
        result = _exec_tool_fn(call["name"], call["args"])
        return {"name": call["name"], "result": result}

    with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as pool:
        outcomes = list(pool.map(_run, calls))

    response_parts = [
        {
            "function_response": {
                "name": o["name"],
                "response": {"result": o["result"]},
            }
        }
        for o in outcomes
    ]
    return history + [{"role": "user", "parts": response_parts}]


def write_tools_node(state: AgentState) -> dict:
    """Execute approved write tools (runs AFTER interrupt is cleared).

    If the user rejected (approved=False), cancel and return a note to the agent.
    """
    calls = state.get("pending_write") or []

    if not state.get("approved"):
        cancel_parts = [
            {
                "function_response": {
                    "name": c["name"],
                    "response": {"result": "[Cancelled by user]"},
                }
            }
            for c in calls
        ]
        return {
            "history": state["history"] + [{"role": "user", "parts": cancel_parts}],
            "pending_write": [],
            "approved": None,
        }

    new_history = _execute_calls(calls, state["history"])
    return {
        "history": new_history,
        "pending_write": [],
        "approved": None,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_agent(state: AgentState) -> Literal["write_tools", "agent", "__end__"]:
    if state.get("pending_write"):
        return "write_tools"          # → interrupt_before here
    if state.get("reply"):
        return "__end__"
    if state.get("iteration", 0) >= MAX_ITER:
        return "__end__"
    return "agent"                    # loop back for tool results


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Build and compile the CollectiveOS LangGraph agent.

    checkpointer: a PostgresSaver (production) or MemorySaver (tests/CLI).
    Pass None to compile without checkpointing (stateless mode).
    """
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("write_tools", write_tools_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent",
        route_agent,
        {
            "write_tools": "write_tools",
            "agent": "agent",
            "__end__": END,
        },
    )
    g.add_edge("write_tools", "agent")

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=["write_tools"],
    )


# ---------------------------------------------------------------------------
# Singleton graph with PostgresSaver
# ---------------------------------------------------------------------------

_graph = None
_checkpointer = None
_exit_stack = None  # keeps PostgresSaver context alive for the process lifetime


def _get_checkpointer():
    global _checkpointer, _exit_stack
    if _checkpointer is not None:
        return _checkpointer
    import contextlib
    import logging

    _exit_stack = contextlib.ExitStack()
    db_url = os.environ.get("DATABASE_URL", "postgresql://assistant:assistant@localhost:5432/assistant")
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        _checkpointer = _exit_stack.enter_context(
            PostgresSaver.from_conn_string(db_url)
        )
        _checkpointer.setup()
        logging.getLogger("collectiveos.agent").info("PostgresSaver checkpointer active")
        return _checkpointer
    except Exception as exc:
        logging.getLogger("collectiveos.agent").warning(
            "PostgresSaver unavailable (%s) — falling back to MemorySaver", exc
        )
        from langgraph.checkpoint.memory import MemorySaver
        _checkpointer = MemorySaver()
        return _checkpointer


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=_get_checkpointer())
    return _graph


# ---------------------------------------------------------------------------
# Convenience wrappers used by api.py and voice_gateway.py
# ---------------------------------------------------------------------------

def run(
    user_message: str,
    system_prompt: str = "",
    thread_id: str = "default",
    history: list[dict] | None = None,
) -> tuple[str, bool]:
    """Run one user message through the LangGraph agent.

    Returns (reply, interrupted):
      reply: the assistant's text response
      interrupted: True if the graph paused before write_tools (needs /chat/approve)
    """
    from src import router

    # Import TOOLS only here to avoid circular imports at module level
    from src.assistant_starter import TOOLS
    active_tools, _ = router.select_tools(user_message, TOOLS)

    # Seed state: prepend prior history + new user message
    prior = history or []
    init_history = prior + [{"role": "user", "parts": [{"text": user_message}]}]

    initial_state: AgentState = {
        "history": init_history,
        "system_prompt": system_prompt,
        "active_tools": active_tools,
        "reply": "",
        "pending_write": [],
        "approved": None,
        "iteration": 0,
    }

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(initial_state, config)

    # Check if we stopped at an interrupt
    state = graph.get_state(config)
    interrupted = bool(state.next)   # non-empty .next means graph is paused

    reply = result.get("reply") or ""
    if interrupted and not reply:
        # Describe what the agent wants to do so the user can approve/reject
        pending = result.get("pending_write") or []
        descriptions = ", ".join(f"{c['name']}({c['args']})" for c in pending)
        reply = f"I'd like to perform: {descriptions}\nShall I go ahead? (yes / no)"

    return reply, interrupted


def approve(thread_id: str, approved: bool) -> str:
    """Resume a paused graph after HITL decision.

    Returns the agent's reply after the write tools execute (or cancel message).
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Update state with the user's decision before resuming
    graph.update_state(config, {"approved": approved})

    result = graph.invoke(None, config)
    return result.get("reply") or ("Done." if approved else "Action cancelled.")
