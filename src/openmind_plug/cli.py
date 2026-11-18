"""Command-line interface for ``openmind-plug``.

Subcommands (``stream``, ``check-stream``, ``record``...) are added by later tasks.
"""

from typing import Annotated

import typer

from openmind_plug import __version__

app = typer.Typer(
    name="openmind-plug",
    help="Read any EEG source and stream it as standard LSL.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"openmind-plug {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """OpenMind Plug: universal EEG plug for LSL."""


@app.command("list-sources")
def list_sources() -> None:
    """List the available EEG sources."""
    typer.echo("No sources available yet.")
