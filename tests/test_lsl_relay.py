"""Tests for the LSL relay source, against simulated third-party streams."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pylsl
import pytest
from typer.testing import CliRunner

from openmind_plug.cli import app
from openmind_plug.inlet import EEGInlet, StreamNotFoundError
from openmind_plug.outlet import EEGOutlet, stream
from openmind_plug.sources import EEGSource
from openmind_plug.sources.lsl_in import LSLSource, LSLSourceError

FS = 250.0
runner = CliRunner()


def third_party_info(
    name: str,
    labels: list[str] | None = None,
    unit: str | None = "volts",
    srate: float = FS,
    with_desc: bool = True,
) -> pylsl.StreamInfo:
    labels = labels or ["Fc3.", "EEG CZ", "c4", "AUX"]
    info = pylsl.StreamInfo(name, "EEG", len(labels), srate, pylsl.cf_double64, f"{name}-id")
    if with_desc:
        desc = info.desc()
        desc.append_child("acquisition").append_child_value("manufacturer", "AcmeEEG")
        desc.append_child("reference").append_child_value("label", "A1")
        channels = desc.append_child("channels")
        for label in labels:
            node = channels.append_child("channel")
            node.append_child_value("label", label)
            if unit:
                node.append_child_value("unit", unit)
            node.append_child_value("type", "MISC" if label == "AUX" else "EEG")
    return info


@contextmanager
def third_party(info: pylsl.StreamInfo) -> Iterator[pylsl.StreamOutlet]:
    """Publish a third-party stream that pushes a ramp (channel i = i * 1e-5)."""
    outlet = pylsl.StreamOutlet(info)
    stop = threading.Event()
    n = info.channel_count()

    def push() -> None:
        while not stop.is_set():
            block = np.tile(np.arange(n, dtype=np.float64) * 1e-5, (25, 1))
            outlet.push_chunk(block)
            time.sleep(0.1)

    thread = threading.Thread(target=push, daemon=True)
    thread.start()
    try:
        yield outlet
    finally:
        stop.set()
        thread.join(timeout=2)


def unique_name() -> str:
    return f"thirdparty-{uuid.uuid4().hex[:8]}"


def read_some(source: EEGSource, n: int = 50) -> np.ndarray:
    blocks, count = [], 0
    deadline = time.monotonic() + 5
    while count < n and time.monotonic() < deadline:
        samples, timestamps = source.read(1000)
        assert timestamps.shape == (samples.shape[0],)
        blocks.append(samples)
        count += samples.shape[0]
        time.sleep(0.05)
    return np.vstack(blocks)


def test_metadata_is_normalized() -> None:
    name = unique_name()
    with third_party(third_party_info(name)):
        source = LSLSource(name=name, timeout=5)
    info = source.info
    assert info.labels == ("FC3", "Cz", "C4", "AUX")
    assert [c.unit for c in info.channels] == ["V"] * 4
    assert [c.type for c in info.channels] == ["EEG", "EEG", "EEG", "MISC"]
    assert info.srate == FS
    assert info.manufacturer == "AcmeEEG"
    assert info.model == name
    assert info.driver == "lsl"
    assert info.reference == "A1"
    assert source.upstream_source_id == f"{name}-id"


def test_reads_samples_in_source_units() -> None:
    name = unique_name()
    with third_party(third_party_info(name)), LSLSource(name=name, timeout=5) as source:
        samples = read_some(source)
    assert samples.shape[1] == 4
    np.testing.assert_allclose(samples[0], [0, 1e-5, 2e-5, 3e-5], rtol=1e-5)


def test_rename_and_montage_select_channels() -> None:
    name = unique_name()
    labels = [f"EXG Channel {i}" for i in range(8)]
    rename = dict(zip(labels, ["Fp1", "FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4"], strict=True))
    with (
        third_party(third_party_info(name, labels=labels, unit="uV")),
        LSLSource(name=name, rename=rename, montage="A", timeout=5) as source,
    ):
        samples = read_some(source)
    assert source.info.labels == ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "Fp1")
    assert source.info.montage == "A"
    # Montage A order picks inputs 1, 2, 3, 4, 5, 6, 7, then 0 (Fp1).
    np.testing.assert_allclose(samples[0] / 1e-5, [1, 2, 3, 4, 5, 6, 7, 0], atol=1e-3)


def test_missing_montage_channel_is_explained() -> None:
    name = unique_name()
    with third_party(third_party_info(name)), pytest.raises(LSLSourceError, match="rename"):
        LSLSource(name=name, montage="A", timeout=5)


def test_stream_without_description_assumes_microvolts(caplog: pytest.LogCaptureFixture) -> None:
    name = unique_name()
    with third_party(third_party_info(name, with_desc=False)):
        source = LSLSource(name=name, timeout=5)
    assert source.info.labels == ("Ch1", "Ch2", "Ch3", "Ch4")
    assert all(c.unit == "uV" for c in source.info.channels)
    assert "assuming microvolts" in caplog.text
    assert source.info.reference == "unknown"


def test_explicit_unit_overrides_description() -> None:
    name = unique_name()
    with third_party(third_party_info(name, unit="volts")):
        source = LSLSource(name=name, unit="mV", timeout=5)
    assert all(c.unit == "mV" for c in source.info.channels)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "openmind-eeg"}, "already a standard stream"),
        ({"name": "x", "unit": "kV"}, "unknown unit"),
    ],
)
def test_rejects_invalid_configuration(kwargs: dict[str, str], message: str) -> None:
    with pytest.raises(LSLSourceError, match=message):
        LSLSource(timeout=0.2, **kwargs)  # type: ignore[arg-type]


def test_rejects_irregular_and_duplicate_streams() -> None:
    name = unique_name()
    with (
        third_party(third_party_info(name, srate=0.0)),
        pytest.raises(LSLSourceError, match="irregular"),
    ):
        LSLSource(name=name, timeout=5)
    name = unique_name()
    with (
        third_party(third_party_info(name, labels=["C3", "c3"])),
        pytest.raises(LSLSourceError, match="duplicate"),
    ):
        LSLSource(name=name, timeout=5)


def test_not_found() -> None:
    with pytest.raises(StreamNotFoundError):
        LSLSource(name=unique_name(), timeout=0.3)


def test_relay_end_to_end_converts_to_microvolts() -> None:
    name = unique_name()
    relay_id = f"openmind-plug-test:{uuid.uuid4()}"
    with third_party(third_party_info(name)):
        source = LSLSource(name=name, timeout=5)
        stop = threading.Event()
        outlet = EEGOutlet(source.info, source_id=relay_id)
        thread = threading.Thread(
            target=stream, args=(source,), kwargs={"stop_event": stop, "outlet": outlet}
        )
        thread.start()
        try:
            with EEGInlet(timeout=5, source_id=relay_id, with_markers=False) as inlet:
                assert inlet.info.labels == ("FC3", "Cz", "C4", "AUX")
                received = np.empty((0, 4), dtype=np.float32)
                deadline = time.monotonic() + 5
                while received.shape[0] < 10 and time.monotonic() < deadline:
                    received = np.vstack([received, inlet.read(timeout=0.2)[0]])
        finally:
            stop.set()
            thread.join(timeout=5)
            outlet.close()
    np.testing.assert_allclose(received[0], [0, 10, 20, 30], rtol=1e-4)  # volts → µV


def test_cli_stream_lsl_relay() -> None:
    name = unique_name()
    with third_party(third_party_info(name)):
        result = runner.invoke(
            app,
            [
                "stream",
                "-s",
                "lsl",
                "--lsl-name",
                name,
                "--duration",
                "0.5",
                "--source-id",
                f"openmind-plug-test:{uuid.uuid4()}",
                "--rename",
                "AUX=EOG1",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "FC3, Cz, C4, EOG1" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "--lsl-name or --lsl-source-id"),
        (["--lsl-name", "x", "--rename", "broken"], "expected old=new"),
        (["--lsl-name", "nothing-here"], "no LSL stream"),
    ],
)
def test_cli_stream_lsl_errors(args: list[str], message: str) -> None:
    import openmind_plug.sources.lsl_in as lsl_in

    original = lsl_in._resolve

    def quick_resolve(*a: object) -> pylsl.StreamInfo:
        return original(a[0], a[1], a[2], 0.3)  # type: ignore[arg-type]

    lsl_in._resolve = quick_resolve  # type: ignore[assignment]
    try:
        result = runner.invoke(app, ["stream", "-s", "lsl", *args])
    finally:
        lsl_in._resolve = original
    assert result.exit_code == 2
    assert message in result.output
