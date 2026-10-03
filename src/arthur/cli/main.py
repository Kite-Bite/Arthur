"""CLI entrypoint: ``arthur`` root application."""

from __future__ import annotations

from pathlib import Path

import typer

from arthur import __version__

app = typer.Typer(
    name="arthur",
    help="Arthur - local-first AI engineering assistant (Ollama + tools + RAG).",
    no_args_is_help=False,
    invoke_without_command=True,
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to config.toml"
    ),
    version: bool = typer.Option(False, "--version", help="Show version and exit"),
) -> None:
    """Arthur CLI. With no subcommand, starts the interactive chat."""
    if version:
        typer.echo(f"arthur {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        from arthur.cli.chat import run_chat

        run_chat(config)


def _register() -> None:
    from arthur.cli import chat, commands

    chat.register(app)
    commands.register(app)


_register()


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
