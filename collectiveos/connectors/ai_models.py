"""
AI Models connector — query external LLMs from within CollectiveOS.

Supported models:
  chatgpt  → OpenAI GPT-4o            (OPENAI_API_KEY)
  claude   → Anthropic Claude Sonnet  (ANTHROPIC_API_KEY — already in env)
  grok     → xAI Grok                 (XAI_API_KEY)
  gemini   → Google Gemini            (GEMINI_API_KEY)

Tools exposed:
  ai_ask(model, prompt)   — query one specific model
  ai_compare(prompt)      — query all configured models in parallel, return side-by-side
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Default model IDs per provider
_DEFAULT_MODELS = {
    "chatgpt": "gpt-4o",
    "claude":  "claude-sonnet-4-6",
    "grok":    "grok-2-latest",
    "gemini":  "gemini-2.0-flash",
}

# Which env key gates each provider
_KEY_MAP = {
    "chatgpt": "OPENAI_API_KEY",
    "claude":  "ANTHROPIC_API_KEY",
    "grok":    "XAI_API_KEY",
    "gemini":  "GEMINI_API_KEY",
}

MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Per-provider callers
# ---------------------------------------------------------------------------

def _ask_openai(prompt: str, system: str = "") -> str:
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=_DEFAULT_MODELS["chatgpt"],
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content or ""


def _ask_claude(prompt: str, system: str = "") -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    kwargs = dict(
        model=_DEFAULT_MODELS["claude"],
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content[0].text


def _ask_grok(prompt: str, system: str = "") -> str:
    # Grok exposes an OpenAI-compatible API
    import openai
    client = openai.OpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=_DEFAULT_MODELS["grok"],
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content or ""


def _ask_gemini(prompt: str, system: str = "") -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    model = genai.GenerativeModel(_DEFAULT_MODELS["gemini"])
    resp = model.generate_content(full_prompt)
    return resp.text


_CALLERS = {
    "chatgpt": _ask_openai,
    "claude":  _ask_claude,
    "grok":    _ask_grok,
    "gemini":  _ask_gemini,
}


def _available_models() -> dict:
    """Return only the models whose API key is set in the environment."""
    return {
        name: caller
        for name, caller in _CALLERS.items()
        if os.environ.get(_KEY_MAP[name])
    }


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def ai_ask(model: str, prompt: str, system: str = "") -> str:
    """
    Query a single AI model and return its response.
    model: one of 'chatgpt', 'claude', 'grok', 'gemini'
    """
    model = model.lower().strip()
    caller = _CALLERS.get(model)
    if caller is None:
        return (
            f"[ERROR: unknown model '{model}'. "
            f"Choose from: {', '.join(_CALLERS)}]"
        )
    if not os.environ.get(_KEY_MAP[model]):
        return f"[ERROR: {_KEY_MAP[model]} not set — configure it in .env to use {model}]"
    try:
        return caller(prompt, system)
    except Exception as exc:
        return f"[ERROR: {type(exc).__name__} — {exc}]"


def ai_compare(prompt: str, system: str = "") -> str:
    """
    Ask all configured AI models the same prompt simultaneously (parallel).
    Returns a side-by-side comparison of all responses.
    """
    available = _available_models()
    if not available:
        return "[ERROR: No AI model API keys configured. Add at least one of OPENAI_API_KEY, XAI_API_KEY, GEMINI_API_KEY to .env]"

    results: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=len(available)) as pool:
        futures = {
            pool.submit(caller, prompt, system): name
            for name, caller in available.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = f"[ERROR: {type(exc).__name__} — {exc}]"

    lines = [f"**Prompt:** {prompt}\n"]
    for name in _CALLERS:           # fixed display order
        if name not in results:
            continue
        label = {
            "chatgpt": "ChatGPT (GPT-4o)",
            "claude":  "Claude (Sonnet)",
            "grok":    "Grok",
            "gemini":  "Gemini",
        }[name]
        lines.append(f"---\n### {label}\n{results[name]}")

    return "\n\n".join(lines)
