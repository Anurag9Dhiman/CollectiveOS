"""
collectiveos CLI — entry point for the installed package.

Commands:
  collectiveos start   — interactive assistant (text)
  collectiveos api     — start the FastAPI server
  collectiveos init    — first-time setup (env check, DB schema)
  collectiveos info    — show platform profile and enabled connectors
"""

import os
import sys

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="collectiveos",
    help="CollectiveOS — portable personal AI assistant framework",
    add_completion=False,
)
console = Console()


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# collectiveos start
# ---------------------------------------------------------------------------

@app.command()
def start():
    """Start the interactive text assistant."""
    _load_env()

    if not os.environ.get("GEMINI_API_KEY"):
        rprint("[red]Error:[/red] GEMINI_API_KEY not set. Add it to your .env file.")
        raise typer.Exit(1)

    # Import here so startup is fast if the key is missing
    import datetime
    from collectiveos import assistant_starter as agent
    from collectiveos import memory, router

    console.print("[bold green]CollectiveOS[/bold green] ready. Type [dim]quit[/dim] to exit.\n")

    cli_history: list[dict] = []

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        past = memory.search(user_input)
        now = datetime.datetime.now(datetime.timezone.utc)
        date_str = now.strftime("%A, %B %-d, %Y, %H:%M UTC")
        system_prompt = f"You are a helpful personal assistant.\nToday is {date_str}."
        if past:
            system_prompt += "\n\nRelevant context from past conversations:\n" + past

        reply = agent.run(user_input, system=system_prompt, history=cli_history)
        console.print(f"[bold cyan]assistant>[/bold cyan] {reply}\n")

        cli_history.append({"role": "user", "content": user_input})
        cli_history.append({"role": "assistant", "content": reply})
        cli_history = cli_history[-20:]

        memory.save(user_input, reply)


# ---------------------------------------------------------------------------
# collectiveos api
# ---------------------------------------------------------------------------

@app.command()
def api(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes"),
):
    """Start the FastAPI server."""
    _load_env()

    if not os.environ.get("API_TOKEN"):
        rprint("[yellow]Warning:[/yellow] API_TOKEN not set — server will reject all requests.")

    import uvicorn
    uvicorn.run("collectiveos.api:app", host=host, port=port, reload=reload)


# ---------------------------------------------------------------------------
# collectiveos init
# ---------------------------------------------------------------------------

@app.command()
def init():
    """First-time setup: check environment and apply DB schema."""
    _load_env()

    console.print("[bold]CollectiveOS — setup check[/bold]\n")

    # Check required env vars
    checks = {
        "GEMINI_API_KEY": "Required — powers the agent loop",
        "DATABASE_URL":   "Required for memory — start Postgres with docker compose up -d",
        "API_TOKEN":      "Required if running collectiveos api",
    }
    all_ok = True
    for key, note in checks.items():
        val = os.environ.get(key)
        if val:
            console.print(f"  [green]✓[/green] {key}")
        else:
            console.print(f"  [red]✗[/red] {key}  [dim]({note})[/dim]")
            all_ok = False

    console.print()

    # Apply DB schema if Postgres is available
    if os.environ.get("DATABASE_URL"):
        schema_path = _find_schema()
        if schema_path:
            try:
                import psycopg2
                conn = psycopg2.connect(os.environ["DATABASE_URL"], connect_timeout=5)
                with open(schema_path) as f:
                    sql = f.read()
                with conn.cursor() as cur:
                    cur.execute(sql)
                conn.commit()
                conn.close()
                console.print("  [green]✓[/green] Database schema applied")
            except Exception as e:
                console.print(f"  [yellow]![/yellow] DB schema skipped: {e}")
        else:
            console.print("  [yellow]![/yellow] schema.sql not found — skipping DB setup")

    console.print()
    if all_ok:
        console.print("[green]All checks passed.[/green] Run [bold]collectiveos start[/bold] to begin.")
    else:
        console.print("Fix the items above, then run [bold]collectiveos init[/bold] again.")


def _find_schema() -> str | None:
    """Find schema.sql relative to the package or CWD."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "schema.sql"),
        os.path.join(os.getcwd(), "schema.sql"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


# ---------------------------------------------------------------------------
# collectiveos info
# ---------------------------------------------------------------------------

@app.command()
def info():
    """Show platform profile and which connectors are enabled on this device."""
    from collectiveos.platform_profile import detect, enabled_connectors

    profile = detect()
    connectors = enabled_connectors(profile)

    console.print(f"\n[bold]CollectiveOS[/bold] — platform info\n")
    console.print(f"  OS:       {profile['os']}")
    console.print(f"  Arch:     {profile['arch']}")
    console.print(f"  Python:   {profile['python']}")
    console.print(f"  Device:   {profile['device_type']}\n")

    table = Table(title="Connectors", show_lines=False)
    table.add_column("Connector", style="cyan")
    table.add_column("Status")
    table.add_column("Note", style="dim")

    for name, status, note in connectors:
        icon = "[green]enabled[/green]" if status else "[dim]unavailable[/dim]"
        table.add_row(name, icon, note)

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# collectiveos voice
# ---------------------------------------------------------------------------

@app.command()
def voice(
    host: str = typer.Option("0.0.0.0", help="Host for the CollectiveOS API server"),
    port: int = typer.Option(8000, help="Port for the CollectiveOS API server"),
    setup: bool = typer.Option(False, "--setup", help="Print VoiceOS setup instructions and exit"),
):
    """
    Start CollectiveOS in voice mode.

    Launches the API server with the /voice/ws WebSocket endpoint active.
    Point the VoiceOS voice-service at ws://<this-host>:<port>/voice/ws.

    VoiceOS repo: https://github.com/Anurag9Dhiman/VoiceOS
    """
    _load_env()

    if setup:
        _print_voice_setup(host, port)
        return

    if not os.environ.get("GEMINI_API_KEY"):
        rprint("[red]Error:[/red] GEMINI_API_KEY not set.")
        raise typer.Exit(1)

    console.print(f"\n[bold green]CollectiveOS[/bold green] — voice mode\n")
    console.print(f"  WebSocket endpoint: [cyan]ws://{host}:{port}/voice/ws[/cyan]")
    console.print(f"  Point VoiceOS voice-service at this address.\n")
    console.print("  Run [bold]collectiveos voice --setup[/bold] for full VoiceOS setup instructions.\n")

    import uvicorn
    uvicorn.run("collectiveos.api:app", host=host, port=port, reload=False)


def _print_voice_setup(host: str, port: int):
    console.print("\n[bold]VoiceOS Integration Setup[/bold]\n")
    console.print("1. Clone VoiceOS:")
    console.print("   [dim]git clone https://github.com/Anurag9Dhiman/VoiceOS[/dim]\n")
    console.print("2. Install the voice-service:")
    console.print("   [dim]cd VoiceOS/services/voice-service && pip install -e .[/dim]\n")
    console.print("3. Set environment variables in VoiceOS/.env:")
    console.print("   [dim]GOOGLE_API_KEY=<your Gemini API key for realtime audio>[/dim]")
    console.print("   [dim]GEMINI_API_KEY=<same key, used for ack/router>[/dim]")
    console.print(f"   [dim]COLLECTIVEOS_WS_URL=ws://{host}:{port}/voice/ws[/dim]\n")
    console.print("4. Start CollectiveOS in voice mode:")
    console.print(f"   [dim]collectiveos voice --host {host} --port {port}[/dim]\n")
    console.print("5. Start the VoiceOS voice-service:")
    console.print("   [dim]cd VoiceOS/services/voice-service && python -m voice_service[/dim]\n")
    console.print("[bold]Flow:[/bold] Mic → Gemini Live (VoiceOS) → WebSocket → CollectiveOS agent → reply → TTS (VoiceOS) → Speaker\n")
    console.print("[bold]Multi-step commands:[/bold] Say something like:")
    console.print('   [italic]"Send a Slack message, then summarise my emails, then add a calendar event"[/italic]')
    console.print("   CollectiveOS will ask for confirmation before each write action.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app()


if __name__ == "__main__":
    main()
