"""Interactive chat REPL and single-shot ``ask`` command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from arthur.agent.types import AgentResult, AgentStep
from arthur.security.confirmation import ConfirmationCallback, ConfirmationRequest, approve
from arthur.services import Services

console = Console()

WELCOME = """\
[bold blue]Arthur[/bold blue] - local-first AI engineering assistant
[dim]Type a request, or /help for commands. /exit or Ctrl-D to quit.[/dim]"""

HELP = """\
[bold]Commands[/bold]
  /help   show this help
  /tools  list available tools
  /exit   quit (also: quit, Ctrl-D)
  anything else is sent to the agent"""


def interactive_confirm(request: ConfirmationRequest) -> bool:
    """Ask the user before a risky operation (used by chat and ask)."""
    style = "red" if request.risk == "high" else "yellow"
    console.print(
        Panel(
            request.describe(),
            title=f"[{style}]Confirmation required[/{style}] ({request.risk} risk)",
            border_style=style,
        )
    )
    return typer.confirm("Proceed?", default=False)


def print_step(step: AgentStep) -> None:
    """Render one agent step as a compact, explainable trace line."""
    palette = {
        "success": "green",
        "denied": "red",
        "declined": "yellow",
        "confirmation_required": "yellow",
        "invalid_args": "red",
        "unknown_tool": "red",
        "timeout": "red",
        "error": "red",
    }
    color = palette.get(step.status, "white")
    line = (
        f"  [dim]{step.index}.[/dim] [bold]{step.tool}[/bold] -> "
        f"[{color}]{step.status}[/{color}] "
        f"[dim]{step.permission} · {step.duration_ms:.0f} ms[/dim]"
    )
    console.print(line)
    if step.thought:
        console.print(f"     [dim]{_clip(step.thought, 160)}[/dim]")
    if step.error and step.status != "success":
        console.print(f"     [{color}]{_clip(step.error, 200)}[/{color}]")


def render_result(
    result: AgentResult,
    *,
    stream: bool,
    on_step: bool = True,
) -> None:
    """Print an agent result: step trace, answer, pending confirmations."""
    if on_step:
        for step in result.steps:
            print_step(step)

    if result.stopped_reason == "confirmation_required" and result.pending_confirmation:
        request = result.pending_confirmation
        console.print(
            Panel(
                request.describe(),
                title="[yellow]Waiting for confirmation[/yellow]",
                border_style="yellow",
            )
        )
        console.print(
            "[dim]Re-run the request with --yes (or confirm interactively in chat) "
            "to approve.[/dim]"
        )
        return

    if result.stopped_reason == "llm_error":
        console.print(f"[red]Error:[/red] {result.error}")
        return

    if not result.answer:
        console.print("[yellow](no answer produced)[/yellow]")
        return

    if stream:
        # Tokens were already streamed to stdout by on_token.
        console.print()
    else:
        console.print(Markdown(result.answer))

    if result.citations:
        console.print("[dim]Sources: " + ", ".join(result.citations) + "[/dim]")
    if result.removed_citations:
        console.print(
            "[yellow]Ignored fabricated citations: "
            + ", ".join(result.removed_citations)
            + "[/yellow]"
        )


def execute_turn(
    services: Services,
    text: str,
    conversation_id: str | None,
    *,
    confirm_handler: ConfirmationCallback,
    show_steps: bool = True,
    stream: bool = True,
) -> str | None:
    """Run one request through the agent with live trace and streaming."""
    streamed = {"started": False}

    def on_token(token: str) -> None:
        if not streamed["started"]:
            console.print()
            streamed["started"] = True
        typer.echo(token, nl=False)

    result = services.agent.run(
        text,
        conversation_id=conversation_id,
        confirm=confirm_handler,
        on_step=print_step if show_steps else None,
        on_token=on_token if stream else None,
    )
    if streamed["started"]:
        typer.echo()
    render_result(result, stream=stream and bool(result.answer), on_step=False)
    if result.stopped_reason == "llm_error":
        raise typer.Exit(code=1)
    return result.conversation_id or conversation_id


def run_chat(config_path: Path | None = None, *, assume_yes: bool = False) -> None:
    """Interactive REPL loop (the default ``arthur`` experience)."""
    services = Services.create(config_path)
    health = services.llm.health()
    console.print(WELCOME)
    if not health.reachable:
        console.print(
            f"[red]LLM unavailable:[/red] {health.detail}. "
            "Chat will fail until the server is running."
        )
    elif not health.model_available:
        console.print(
            f"[yellow]Model {health.model!r} not pulled yet[/yellow] "
            f"- run: ollama pull {health.model}"
        )
    if (
        health.reachable
        and not health.embedding_available
        and services.embedder.name.startswith("ollama:")
    ):
        console.print(
            f"[yellow]Embedding model {health.embedding_model!r} not pulled yet[/yellow] "
            f"- RAG will fail until you run: ollama pull {health.embedding_model}"
        )

    confirm_handler = approve if assume_yes else interactive_confirm
    conversation_id: str | None = None
    try:
        while True:
            try:
                text = console.input("\n[bold blue]Arthur >[/bold blue] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not text:
                continue
            lowered = text.lower()
            if lowered in {"exit", "quit", "/exit", "/quit"}:
                break
            if lowered in {"/help", "help"}:
                console.print(HELP)
                continue
            if lowered in {"/tools", "tools"}:
                _print_tools_table(services)
                continue
            try:
                conversation_id = execute_turn(
                    services, text, conversation_id, confirm_handler=confirm_handler
                )
            except typer.Exit:
                console.print(
                    "[red]Request failed (see error above).[/red] Check that Ollama is running."
                )
    finally:
        services.close()


def _print_tools_table(services: Services) -> None:
    from rich.table import Table

    table = Table(title="Registered tools")
    table.add_column("Tool", style="bold")
    table.add_column("Permission")
    table.add_column("Description")
    for tool in services.registry.list():
        table.add_row(tool.name, tool.permission.name, _clip(tool.description, 70))
    console.print(table)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def register(app: typer.Typer) -> None:
    """Attach chat/ask commands to the root app."""

    @app.command("chat")
    def chat_cmd(
        config: Path | None = typer.Option(None, "--config", "-c", help="Config file"),
        yes: bool = typer.Option(
            False, "--yes", "-y", help="Auto-confirm risky operations (audited)"
        ),
    ) -> None:
        """Start the interactive chat REPL."""
        run_chat(config, assume_yes=yes)

    @app.command("ask")
    def ask_cmd(
        question: str = typer.Argument(..., help="One-shot request"),
        config: Path | None = typer.Option(None, "--config", "-c", help="Config file"),
        yes: bool = typer.Option(
            False, "--yes", "-y", help="Auto-confirm risky operations (audited)"
        ),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="Hide the step trace"),
    ) -> None:
        """Answer a single request and exit."""
        services = Services.create(config)
        try:
            execute_turn(
                services,
                question,
                None,
                confirm_handler=approve if yes else interactive_confirm,
                show_steps=not quiet,
                stream=not quiet,
            )
        finally:
            services.close()
