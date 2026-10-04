"""CLI subcommands: system, memory, documents, tools, logs, config, doctor, serve."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from arthur.services import Services

console = Console()


# --- helpers -------------------------------------------------------------------


def _outcome_data(outcome: Any) -> None:
    """Pretty-print an execution outcome, exiting non-zero on failure."""
    if outcome.status != "success":
        console.print(f"[red]{outcome.status}:[/red] {outcome.record.error}")
        raise typer.Exit(code=1)
    data = outcome.result.data if outcome.result else None
    if data is None:
        console.print(outcome.result.summary if outcome.result else "")
        return
    if outcome.record.tool_name == "process_list":
        _print_process_table(data)
        return
    console.print_json(json.dumps(data, default=str))


def _print_process_table(data: dict[str, Any]) -> None:
    table = Table(title="Processes")
    for column in ("pid", "name", "cpu%", "memory", "status", "command"):
        table.add_column(column)
    for proc in data.get("processes", []):
        table.add_row(
            str(proc.get("pid")),
            str(proc.get("name")),
            f"{proc.get('cpu_percent', 0):.1f}",
            str(proc.get("memory_rss")),
            str(proc.get("status")),
            str(proc.get("command"))[:60],
        )
    console.print(table)


# --- system --------------------------------------------------------------------

system_app = typer.Typer(help="System information (safe, no shell required)")

_SYSTEM_MAP = {
    "info": "system_info",
    "cpu": "cpu_info",
    "mem": "memory_info",
    "disk": "disk_usage",
    "ps": "process_list",
    "net": "network_info",
    "uptime": "uptime",
}


def _run_system_tool(command: str, config: Path | None) -> None:
    services = Services.create(config)
    try:
        args: dict[str, Any] = {}
        if command == "disk":
            args = {"mount": "/"}
        outcome = services.executor.execute(_SYSTEM_MAP[command], args)
        _outcome_data(outcome)
    finally:
        services.close()


for _name in _SYSTEM_MAP:

    def _make(name: str) -> Callable[..., None]:
        def _cmd(
            config: Path | None = typer.Option(None, "--config", "-c", help="Config file"),
        ) -> None:
            _run_system_tool(name, config)

        _cmd.__name__ = f"system_{name}"
        return _cmd

    system_app.command(_name)(_make(_name))


# --- memory --------------------------------------------------------------------

memory_app = typer.Typer(help="Long-term memory: add, search, list, delete")


@memory_app.command("add")
def memory_add(
    text: str = typer.Argument(..., help="Fact to remember"),
    category: str = typer.Option("general", "--category", "-g"),
    importance: float = typer.Option(0.5, "--importance", min=0.0, max=1.0),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Store a memory."""
    services = Services.create(config)
    try:
        item = services.memory.add(text, category=category, importance=importance)
        console.print(f"[green]stored[/green] memory #{item.id} in category '{item.category}'")
    finally:
        services.close()


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(...),
    limit: int = typer.Option(5, "--limit", "-n", min=1, max=100),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Search memories by keywords."""
    services = Services.create(config)
    try:
        items = services.memory.search(query, limit=limit)
        _print_memories(items)
    finally:
        services.close()


@memory_app.command("list")
def memory_list(
    limit: int = typer.Option(50, "--limit", "-n", min=1, max=500),
    category: str | None = typer.Option(None, "--category", "-g"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """List stored memories (newest first)."""
    services = Services.create(config)
    try:
        items = services.memory.list(limit=limit, category=category)
        _print_memories(items)
        console.print(f"[dim]{len(items)} of {services.memory.count()} memories[/dim]")
    finally:
        services.close()


@memory_app.command("delete")
def memory_delete(
    memory_id: int | None = typer.Argument(None, help="Memory id to delete"),
    all: bool = typer.Option(False, "--all", help="Delete every memory"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Delete a memory (or all of them with --all)."""
    if all is False and memory_id is None:
        console.print("[red]provide a memory id or --all[/red]")
        raise typer.Exit(code=2)
    services = Services.create(config)
    try:
        if all:
            count = services.memory.delete_all()
            console.print(f"[yellow]deleted {count} memories[/yellow]")
        else:
            assert memory_id is not None
            if services.memory.delete(memory_id):
                console.print(f"[green]deleted[/green] memory #{memory_id}")
            else:
                console.print(f"[red]no such memory:[/red] {memory_id}")
                raise typer.Exit(code=1)
    finally:
        services.close()


def _print_memories(items: list) -> None:
    if not items:
        console.print("[dim]no memories found[/dim]")
        return
    table = Table(title="Memories")
    table.add_column("id", justify="right")
    table.add_column("category")
    table.add_column("imp", justify="right")
    table.add_column("created")
    table.add_column("content")
    for item in items:
        table.add_row(
            str(item.id),
            item.category,
            f"{item.importance:.2f}",
            item.created_at.strftime("%Y-%m-%d"),
            item.content[:90],
        )
    console.print(table)


# --- documents -----------------------------------------------------------------

documents_app = typer.Typer(help="RAG: index and search local documents")


@documents_app.command("index")
def documents_index(
    path: Path = typer.Argument(..., help="File or directory to index"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Index documents into the local vector store."""
    services = Services.create(config)
    try:
        outcome = services.executor.execute(
            "index_documents", {"path": str(path), "recursive": recursive}
        )
        _outcome_data(outcome)
    finally:
        services.close()


@documents_app.command("search")
def documents_search(
    query: str = typer.Argument(...),
    limit: int = typer.Option(5, "--limit", "-n", min=1, max=20),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Semantic search over indexed documents."""
    services = Services.create(config)
    try:
        outcome = services.executor.execute("search_documents", {"query": query, "limit": limit})
        if outcome.status != "success":
            console.print(f"[red]{outcome.status}:[/red] {outcome.record.error}")
            raise typer.Exit(code=1)
        data = outcome.result.data if outcome.result else {}
        for item in data.get("results", []):
            console.print(
                f"[bold]{item['source']}[/bold] [dim](chunk {item['chunk_index']}, "
                f"score {item['score']})[/dim]"
            )
            console.print(f"  {_clip(item['text'], 400)}")
            console.print()
    finally:
        services.close()


# --- tools ---------------------------------------------------------------------

tools_app = typer.Typer(help="Inspect registered tools")


@tools_app.command("list")
def tools_list(
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """List every registered tool with its permission level."""
    services = Services.create(config)
    try:
        table = Table(title=f"Tools ({len(services.registry)})")
        table.add_column("Tool", style="bold")
        table.add_column("Permission")
        table.add_column("Description")
        for tool in services.registry.list():
            table.add_row(tool.name, tool.permission.name, _clip(tool.description, 70))
        console.print(table)
        if not services.config.security.default_shell_access:
            console.print(
                "[dim]run_command is present but denied while "
                "security.default_shell_access = false[/dim]"
            )
    finally:
        services.close()


@tools_app.command("show")
def tools_show(
    name: str = typer.Argument(..., help="Tool name"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Show a tool's description, permission and input schema."""
    services = Services.create(config)
    try:
        tool = services.registry.get(name)
        if tool is None:
            console.print(f"[red]unknown tool:[/red] {name}")
            raise typer.Exit(code=1)
        console.print(f"[bold]{tool.name}[/bold] - {tool.permission.name} (risk: {tool.risk})")
        console.print(tool.description)
        console.print_json(json.dumps(tool.schema()["parameters"], default=str))
    finally:
        services.close()


# --- standalone commands -------------------------------------------------------


def _logs_cmd(
    limit: int = typer.Option(20, "--limit", "-n", min=1, max=500),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Show recent tool executions (the audit trail)."""
    services = Services.create(config)
    try:
        rows = services.executions.recent(limit)
        table = Table(title=f"Recent activity ({len(rows)})")
        table.add_column("time")
        table.add_column("tool")
        table.add_column("status")
        table.add_column("perm")
        table.add_column("ms", justify="right")
        table.add_column("request")
        table.add_column("error")
        for row in rows:
            table.add_row(
                row.created_at.strftime("%m-%d %H:%M:%S"),
                row.tool_name,
                _status_cell(row.status),
                row.permission + ("*" if row.confirmed else ""),
                f"{row.duration_ms:.0f}",
                _clip(row.request or "", 40),
                _clip(row.error or "", 40),
            )
        console.print(table)
        console.print("[dim]* = confirmed by the user[/dim]")
    finally:
        services.close()


def _status_cell(status: str) -> str:
    color = {
        "success": "green",
        "denied": "red",
        "declined": "yellow",
        "confirmation_required": "yellow",
        "error": "red",
        "timeout": "red",
    }.get(status, "white")
    return f"[{color}]{status}[/{color}]"


def _config_show(
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Show the effective configuration (secrets redacted)."""
    services = Services.create(config)
    try:
        data = services.config.model_dump(mode="json")
        if data.get("api", {}).get("token"):
            data["api"]["token"] = "***redacted***"
        console.print_json(json.dumps(data, default=str))
        console.print("[dim]precedence: defaults < config.toml < .env < environment[/dim]")
    finally:
        services.close()


def _doctor(
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Check LLM, storage and index health."""
    services = Services.create(config)
    try:
        health = services.health()
        rows = [
            (
                "ollama",
                "ok" if health["llm_reachable"] else "UNAVAILABLE",
                str(health["llm_detail"]),
            ),
            ("model", "ok" if health["model_available"] else "MISSING", str(health["model"])),
            ("tools", "ok", f"{health['tools']} registered"),
            ("memory", "ok", f"{health['memories']} entries"),
            ("documents", "ok", f"{health['indexed_documents']} indexed"),
            ("audit", "ok", f"{health['audit_records']} records"),
        ]
        table = Table(title="Arthur doctor")
        table.add_column("check")
        table.add_column("status")
        table.add_column("detail")
        for check, status, detail in rows:
            style = "green" if status in {"ok"} else "red"
            table.add_row(check, f"[{style}]{status}[/{style}]", detail)
        console.print(table)
        if not health["llm_reachable"]:
            raise typer.Exit(code=1)
    finally:
        services.close()


def _serve(
    host: str | None = typer.Option(None, "--host", help="Bind address"),
    port: int | None = typer.Option(None, "--port", help="Port"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Serve the FastAPI backend."""
    import uvicorn

    from arthur.api.app import create_app

    services = Services.create(config)
    bind_host = host or services.config.api.host
    bind_port = port or services.config.api.port
    if not services.config.api.token:
        console.print(
            "[yellow]No ARTHUR_API_TOKEN configured - the API is unauthenticated "
            "(loopback only).[/yellow]"
        )
    uvicorn.run(create_app(services), host=bind_host, port=bind_port)


def register(app: typer.Typer) -> None:
    """Attach sub-apps and standalone commands to the root app."""
    app.add_typer(system_app, name="system")
    app.add_typer(memory_app, name="memory")
    app.add_typer(documents_app, name="documents")
    app.add_typer(tools_app, name="tools")
    app.command("logs")(_logs_cmd)
    app.command("config-show")(_config_show)
    app.command("doctor")(_doctor)
    app.command("serve")(_serve)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
