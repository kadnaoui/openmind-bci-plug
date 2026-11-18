"""Command-line interface for ``openmind-plug``."""

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


@app.command("check-stream")
def check_stream(
    timeout: Annotated[
        float, typer.Option(help="Seconds to wait for the stream to appear.", min=0.1)
    ] = 5.0,
    duration: Annotated[
        float, typer.Option(help="Seconds of data to receive and check.", min=0.5)
    ] = 2.0,
    source_id: Annotated[
        str | None, typer.Option(help="Check the stream with this LSL source_id.")
    ] = None,
) -> None:
    """Check that a live EEG stream follows the OpenMind stream specification.

    Exits with code 1 if the stream breaks the specification, 2 if none is found.
    """
    from openmind_plug.inlet import StreamNotFoundError
    from openmind_plug.inlet import check_stream as run_check
    from openmind_plug.spec import SPEC_VERSION

    try:
        report = run_check(timeout=timeout, duration=duration, source_id=source_id)
    except StreamNotFoundError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"Stream {report.source_id} (specification {SPEC_VERSION})")
    if report.info is not None:
        info = report.info
        typer.echo(f"  {info.n_channels} channels @ {info.srate:g} Hz: {', '.join(info.labels)}")
        typer.echo(
            f"  {info.manufacturer} / {info.model} / driver {info.driver}, "
            f"reference {info.reference}, montage {info.montage or '-'}"
        )
    if report.n_samples:
        typer.echo(
            f"  received {report.n_samples} samples ({report.effective_srate:.1f} Hz), "
            f"{report.n_markers} markers"
        )
    for warning in report.warnings:
        typer.secho(f"  warning: {warning}", fg=typer.colors.YELLOW)
    for problem in report.errors:
        typer.secho(f"  error: {problem}", fg=typer.colors.RED)
    if report.ok:
        typer.secho("OK: the stream follows the specification.", fg=typer.colors.GREEN)
    else:
        typer.secho(f"FAILED: {len(report.errors)} problem(s).", fg=typer.colors.RED)
        raise typer.Exit(code=1)
