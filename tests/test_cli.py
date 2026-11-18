"""Tests for the ``openmind-plug`` command line (stream, list-sources, keyboard)."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openmind_plug.cli import SOURCES, SourceName, _source_status, _start_keyboard, app
from openmind_plug.keyboard import run_key_loop
from openmind_plug.sources.synthetic import SyntheticSource

runner = CliRunner()


def unique_id() -> str:
    return f"openmind-plug-test:{uuid.uuid4()}"


def test_list_sources_shows_every_source() -> None:
    result = runner.invoke(app, ["list-sources"])
    assert result.exit_code == 0
    for name in SourceName:
        assert name.value in result.output
    assert "planned" in result.output


def test_source_status() -> None:
    assert _source_status(SOURCES[SourceName.synthetic]) == "available"
    assert _source_status(SOURCES[SourceName.lsl]) == "planned"


@pytest.mark.parametrize("command", ["stream", "check-stream", "list-sources"])
def test_every_command_has_help(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_stream_synthetic_for_a_duration() -> None:
    source_id = unique_id()
    result = runner.invoke(
        app,
        [
            "stream",
            "-s",
            "synthetic",
            "--duration",
            "0.5",
            "--no-keyboard",
            "--seed",
            "1",
            "--source-id",
            source_id,
            "--montage",
            "b",
            "--blink-rate",
            "60",
        ],
    )
    assert result.exit_code == 0, result.output
    assert f"Streaming {source_id}" in result.output
    assert "C3, Cz, C4, CP3, CP4, Fp1, O1, O2" in result.output
    assert "Stopped after" in result.output


def test_stream_replay_file_until_the_end(tmp_path: Path) -> None:
    mne = pytest.importorskip("mne")
    import numpy as np

    raw = mne.io.RawArray(
        np.zeros((3, 500)), mne.create_info(["C3", "Cz", "C4"], 250.0, "eeg"), verbose=False
    )
    path = tmp_path / "short_raw.fif"
    raw.save(path, verbose=False)
    result = runner.invoke(
        app,
        [
            "stream",
            "-s",
            "replay",
            "--file",
            str(path),
            "--speed",
            "20",
            "--no-keyboard",
            "--source-id",
            unique_id(),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "(end of recording): 500 samples" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["-s", "lsl"], "not available: planned"),
        (["-s", "synthetic", "--montage", "Z"], "unknown montage"),
        (["-s", "synthetic", "--srate", "50"], "srate"),
        (["-s", "synthetic", "--line-freq", "55"], "line_freq_hz"),
        (["-s", "replay"], "exactly one of --dataset or --file"),
        (["-s", "replay", "--dataset", "nope"], "unknown dataset"),
    ],
)
def test_stream_reports_usage_errors(args: list[str], message: str) -> None:
    result = runner.invoke(app, ["stream", *args, "--no-keyboard"])
    assert result.exit_code == 2
    assert message in result.output


def test_stream_replay_missing_channels(tmp_path: Path) -> None:
    mne = pytest.importorskip("mne")
    import numpy as np

    raw = mne.io.RawArray(
        np.zeros((2, 250)), mne.create_info(["C3", "C4"], 250.0, "eeg"), verbose=False
    )
    path = tmp_path / "two_raw.fif"
    raw.save(path, verbose=False)
    result = runner.invoke(
        app, ["stream", "-s", "replay", "--file", str(path), "-m", "A", "--no-keyboard"]
    )
    assert result.exit_code == 2
    assert "omit --montage" in result.output


# --- Keyboard -------------------------------------------------------------------


def test_run_key_loop_calls_handler_until_false() -> None:
    keys = iter(["a", None, "b", "q", "z"])
    seen: list[str] = []
    stop = threading.Event()

    def on_key(key: str) -> bool:
        seen.append(key)
        return key != "q"

    run_key_loop(lambda timeout: next(keys), on_key, stop)
    assert seen == ["a", "b", "q"]
    assert stop.is_set()


def test_keyboard_drives_synthetic_source() -> None:
    source = SyntheticSource(seed=0)
    keys = iter(["a", "e", "x", "q"])
    done = threading.Event()

    def read(timeout: float) -> str | None:
        key = next(keys, None)
        if key is None:
            done.wait(timeout)
        return key

    stop = threading.Event()
    _start_keyboard(source, read, stop)
    assert stop.wait(5), "q should stop the stream"
    assert source.state == "left"
    assert source.eyes == "closed"
