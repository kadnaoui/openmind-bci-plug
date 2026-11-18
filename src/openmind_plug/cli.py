"""Command-line interface for ``openmind-plug``."""

from __future__ import annotations

import importlib.util
import threading
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from openmind_plug import __version__
from openmind_plug.sources.base import EEGSource

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


class SourceName(StrEnum):
    """Sources accepted by ``stream --source``."""

    synthetic = "synthetic"
    replay = "replay"
    lsl = "lsl"
    brainflow = "brainflow"
    ironbci = "ironbci"


@dataclass(frozen=True)
class _SourceEntry:
    description: str
    extra: str | None = None  # optional dependency group, if any
    module: str | None = None  # module that must be importable
    planned: bool = False


SOURCES: dict[SourceName, _SourceEntry] = {
    SourceName.synthetic: _SourceEntry("Realistic synthetic EEG with keyboard-controlled state"),
    SourceName.replay: _SourceEntry(
        "Recorded files (any MNE format) or public datasets, in real time"
    ),
    SourceName.lsl: _SourceEntry("Relay of a third-party LSL stream", planned=True),
    SourceName.brainflow: _SourceEntry("Any BrainFlow-supported board", "brainflow", planned=True),
    SourceName.ironbci: _SourceEntry(
        "IronBCI-8 headset over BLE (experimental)", "ironbci", planned=True
    ),
}


def _source_status(entry: _SourceEntry) -> str:
    if entry.planned:
        return "planned"
    if entry.module and importlib.util.find_spec(entry.module) is None:
        return f'needs: pip install "openmind-plug[{entry.extra}]"'
    return "available"


@app.command("list-sources")
def list_sources() -> None:
    """List the EEG sources and whether they can be used."""
    width = max(len(name.value) for name in SOURCES)
    for name, entry in SOURCES.items():
        status = _source_status(entry)
        color = typer.colors.GREEN if status == "available" else typer.colors.YELLOW
        typer.echo(f"{name.value:<{width}}  ", nl=False)
        typer.secho(f"{status:<10}", fg=color, nl=False)
        note = ""
        if name is SourceName.replay and not _datasets_available():
            note = f" (datasets need: {_REPLAY_INSTALL})"
        typer.echo(f"  {entry.description}{note}")


_REPLAY_INSTALL = 'pip install "openmind-plug[replay]"'


def _datasets_available() -> bool:
    """Whether MOABB, needed to download public datasets, is installed."""
    return importlib.util.find_spec("moabb") is not None


_SYNTH = "Synthetic source"
_REPLAY = "Replay source"


@app.command("stream")
def stream_command(
    source: Annotated[
        SourceName, typer.Option("--source", "-s", help="Source to stream.", show_default=False)
    ],
    montage: Annotated[
        str | None,
        typer.Option(
            "--montage",
            "-m",
            help="Montage (A, B). Default: A for synthetic, every EEG channel for replay.",
            show_default=False,
        ),
    ] = None,
    srate: Annotated[float, typer.Option(help="Sampling rate in Hz.", min=1)] = 250.0,
    duration: Annotated[
        float | None,
        typer.Option(help="Stop after this many seconds (default: until Ctrl+C).", min=0.1),
    ] = None,
    source_id: Annotated[
        str | None, typer.Option(help="LSL source_id (default: openmind-plug:<driver>:<model>).")
    ] = None,
    keyboard: Annotated[
        bool | None,
        typer.Option(
            "--keyboard/--no-keyboard",
            help="Control the synthetic source from the keyboard (default: on in a terminal).",
            show_default=False,
        ),
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Random seed.", rich_help_panel=_SYNTH)] = None,
    blink_rate: Annotated[
        float, typer.Option(help="Random blinks per minute.", min=0, rich_help_panel=_SYNTH)
    ] = 0.0,
    muscle_rate: Annotated[
        float,
        typer.Option(help="Random muscle bursts per minute.", min=0, rich_help_panel=_SYNTH),
    ] = 0.0,
    line_noise: Annotated[
        float,
        typer.Option(help="Power-line noise amplitude in µV.", min=0, rich_help_panel=_SYNTH),
    ] = 0.0,
    line_freq: Annotated[
        int, typer.Option(help="Power-line frequency (50 or 60 Hz).", rich_help_panel=_SYNTH)
    ] = 50,
    dataset: Annotated[
        str | None,
        typer.Option(help="Public dataset (bnci2014_001, physionetmi).", rich_help_panel=_REPLAY),
    ] = None,
    subject: Annotated[
        int, typer.Option(help="Dataset subject number.", min=1, rich_help_panel=_REPLAY)
    ] = 1,
    session: Annotated[
        str | None,
        typer.Option(help="Dataset session (default: first).", rich_help_panel=_REPLAY),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option(
            help="Recording to replay (FIF, EDF, BDF...), instead of a dataset.",
            exists=True,
            dir_okay=False,
            rich_help_panel=_REPLAY,
        ),
    ] = None,
    speed: Annotated[
        float,
        typer.Option(help="Playback speed (1 = real time).", min=0.01, rich_help_panel=_REPLAY),
    ] = 1.0,
    loop: Annotated[
        bool, typer.Option("--loop", help="Restart at the end.", rich_help_panel=_REPLAY)
    ] = False,
) -> None:
    """Stream an EEG source as the standard OpenMind LSL streams.

    Examples:

      openmind-plug stream --source synthetic --montage A

      openmind-plug stream --source replay --dataset bnci2014_001 --subject 1
    """
    from openmind_plug.keyboard import key_reader, stdin_is_terminal
    from openmind_plug.outlet import EEGOutlet, stream

    entry = SOURCES[source]
    status = _source_status(entry)
    if status != "available":
        _fail(f"source {source.value!r} is not available: {status}")

    built = _build_source(
        source,
        montage=montage,
        srate=srate,
        seed=seed,
        blink_rate=blink_rate,
        muscle_rate=muscle_rate,
        line_noise=line_noise,
        line_freq=line_freq,
        dataset=dataset,
        subject=subject,
        session=session,
        file=file,
        speed=speed,
        loop=loop,
    )
    info = built.info
    use_keyboard = source is SourceName.synthetic and (
        keyboard if keyboard is not None else stdin_is_terminal()
    )

    with EEGOutlet(info, source_id=source_id) as outlet, ExitStack() as stack:
        typer.secho(f"Streaming {outlet.source_id}", bold=True)
        typer.echo(f"  {info.n_channels} channels @ {info.srate:g} Hz: {', '.join(info.labels)}")
        stop = threading.Event()
        if use_keyboard:
            _start_keyboard(built, stack.enter_context(key_reader()), stop)
        else:
            typer.echo("  Press Ctrl+C to stop.")
        stats = stream(built, duration=duration, stop_event=stop, outlet=outlet)
        stop.set()

    ending = " (end of recording)" if getattr(built, "finished", False) else ""
    typer.echo(
        f"Stopped after {stats.elapsed:.1f} s{ending}: "
        f"{stats.n_samples} samples, {stats.n_markers} markers."
    )


@app.command("record")
def record_command(
    subject: Annotated[
        str,
        typer.Option(
            help="Pseudonymous subject id (P01, P02...), never a name.", show_default=False
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            help="Output file, ending with _raw.fif "
            "(default: $OPENMIND_DATA_DIR/recordings/<subject>/<subject>_<date>_raw.fif).",
            dir_okay=False,
            show_default=False,
        ),
    ] = None,
    duration: Annotated[
        float | None,
        typer.Option(help="Stop after this many seconds (default: until Ctrl+C).", min=0.1),
    ] = None,
    timeout: Annotated[
        float, typer.Option(help="Seconds to wait for the stream to appear.", min=0.1)
    ] = 5.0,
    source_id: Annotated[
        str | None, typer.Option(help="Record the stream with this LSL source_id.")
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing file.")] = False,
) -> None:
    """Record the standard LSL stream (EEG and markers) to a FIF file.

    Example:

      openmind-plug record --subject P01 --duration 60
    """
    from openmind_plug.inlet import StreamNotFoundError
    from openmind_plug.recorder import default_path, record_stream, validate_subject

    try:
        validate_subject(subject)
    except ValueError as error:
        _fail(str(error))
    target = out or default_path(subject)
    if not target.name.endswith("_raw.fif"):
        _fail(f"recording file names must end with '_raw.fif', got {target.name}")
    if target.exists() and not overwrite:
        _fail(f"{target} already exists (use --overwrite to replace it)")
    if _inside_git_repository(target):
        typer.secho(
            f"Warning: {target.parent} is inside a Git repository; EEG recordings are "
            "health data and must never be committed.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    try:
        typer.echo("Waiting for the stream...")
        recorder = record_stream(subject, duration=duration, timeout=timeout, source_id=source_id)
    except StreamNotFoundError as error:
        _fail(str(error))
    if recorder.n_samples == 0:
        _fail("no sample received; nothing was saved")
    path = recorder.save(target, overwrite=overwrite)
    typer.echo(
        f"Saved {recorder.duration:.1f} s ({recorder.n_samples} samples, "
        f"{recorder.n_markers} markers) to {path}"
    )


def _inside_git_repository(path: Path) -> bool:
    """Whether ``path`` would be written inside a Git working tree."""
    return any((parent / ".git").exists() for parent in path.resolve().parents)


def _build_source(
    source: SourceName,
    *,
    montage: str | None,
    srate: float,
    seed: int | None,
    blink_rate: float,
    muscle_rate: float,
    line_noise: float,
    line_freq: int,
    dataset: str | None,
    subject: int,
    session: str | None,
    file: Path | None,
    speed: float,
    loop: bool,
) -> EEGSource:
    from pydantic import ValidationError

    from openmind_plug.montage import Montage, get_montage

    resolved: Montage | None = None
    if montage is not None:
        try:
            resolved = get_montage(montage)
        except KeyError as error:
            _fail(str(error.args[0]))

    if source is SourceName.synthetic:
        from openmind_plug.sources.synthetic import SyntheticConfig, SyntheticSource

        try:
            config = SyntheticConfig(
                blink_rate_per_min=blink_rate,
                muscle_rate_per_min=muscle_rate,
                line_noise_uv=line_noise,
                line_freq_hz=line_freq,  # type: ignore[arg-type]  # validated by pydantic
            )
            return SyntheticSource(montage=resolved or "A", srate=srate, seed=seed, config=config)
        except (ValidationError, ValueError) as error:
            _fail(str(error))

    from openmind_plug.sources.replay import MissingChannelsError, ReplaySource

    if (dataset is None) == (file is None):
        _fail("replay needs exactly one of --dataset or --file")
    try:
        if file is not None:
            return ReplaySource.from_file(
                file, montage=resolved, srate=srate, speed=speed, loop=loop
            )
        if not _datasets_available():
            _fail(f"replaying public datasets needs MOABB: {_REPLAY_INSTALL}")
        typer.echo(f"Loading {dataset} subject {subject} (downloaded on first use)...")
        return ReplaySource.from_dataset(
            str(dataset),
            subject=subject,
            session=session,
            montage=resolved,
            srate=srate,
            speed=speed,
            loop=loop,
        )
    except (MissingChannelsError, KeyError, ValueError, ImportError) as error:
        _fail(str(error.args[0]) if error.args else str(error))


def _start_keyboard(
    source: EEGSource, read: Callable[[float], str | None], stop: threading.Event
) -> None:
    """Read keys in a background thread and apply them to the synthetic source."""
    from openmind_plug.keyboard import run_key_loop
    from openmind_plug.sources.synthetic import KEY_BINDINGS, SyntheticSource

    assert isinstance(source, SyntheticSource)
    bindings = ", ".join(f"{key}: {action}" for key, action in KEY_BINDINGS.items())
    typer.echo(f"  Keys: {bindings}, q: quit.")

    def on_key(key: str) -> bool:
        if key.lower() == "q":
            return False
        if source.handle_key(key):
            typer.echo(f"  state: {source.state}, eyes: {source.eyes}")
        return True

    threading.Thread(target=run_key_loop, args=(read, on_key, stop), daemon=True).start()


def _fail(message: str) -> NoReturn:
    """Print an error and exit with code 2 (usage error)."""
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=2)


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
