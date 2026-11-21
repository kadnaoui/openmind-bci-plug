"""Outlet → inlet tests over a real (local) LSL network.

Each test publishes under a unique ``source_id`` so tests never pick up each
other's streams, nor a stream running elsewhere on the machine.
"""

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
from openmind_plug.inlet import EEGInlet, StreamNotFoundError, check_stream
from openmind_plug.montage import get_montage
from openmind_plug.outlet import EEGOutlet, stream
from openmind_plug.sources import ChannelInfo, Marker, StreamInfo
from openmind_plug.sources.synthetic import SyntheticSource

INFO = StreamInfo(
    srate=250.0,
    channels=get_montage("A").channels(),
    manufacturer="OpenMind",
    model="test",
    driver="test",
    reference="Cz",
    montage="A",
)


def unique_id() -> str:
    return f"openmind-plug-test:{uuid.uuid4()}"


def read_until(inlet: EEGInlet, n: int, timeout: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    chunks, stamps = [], []
    deadline = time.monotonic() + timeout
    while sum(len(s) for s in stamps) < n and time.monotonic() < deadline:
        samples, timestamps = inlet.read(timeout=0.2)
        chunks.append(samples)
        stamps.append(timestamps)
    return np.vstack(chunks), np.concatenate(stamps)


@contextmanager
def streaming(source: SyntheticSource, source_id: str) -> Iterator[EEGOutlet]:
    """Publish ``source`` in a background thread for the duration of the block."""
    stop = threading.Event()
    outlet = EEGOutlet(source.info, source_id=source_id)
    thread = threading.Thread(
        target=stream, args=(source,), kwargs={"stop_event": stop, "outlet": outlet}
    )
    thread.start()
    try:
        yield outlet
    finally:
        stop.set()
        thread.join(timeout=5)
        outlet.close()


def test_round_trip_metadata_and_samples_without_loss() -> None:
    source_id = unique_id()
    with (
        EEGOutlet(INFO, source_id=source_id) as outlet,
        EEGInlet(timeout=5, source_id=source_id) as inlet,
    ):
        assert inlet.info == INFO
        assert inlet.source_id == source_id
        assert inlet.has_markers
        assert outlet.wait_for_consumers(5)

        rng = np.random.default_rng(0)
        sent = rng.normal(0, 20, (1000, 8)).astype(np.float32)
        t0 = pylsl.local_clock()
        times = t0 + np.arange(1000) / 250.0
        for start in range(0, 1000, 100):
            outlet.push(sent[start : start + 100], times[start : start + 100])
        received, stamps = read_until(inlet, 1000)

    np.testing.assert_array_equal(received, sent)
    np.testing.assert_allclose(stamps, times, atol=1e-3)


def test_markers_round_trip() -> None:
    source_id = unique_id()
    with (
        EEGOutlet(INFO, source_id=source_id) as outlet,
        EEGInlet(timeout=5, source_id=source_id, markers_from=[f"{source_id}:markers"]) as inlet,
    ):
        time.sleep(0.5)  # let the markers inlet connect
        now = pylsl.local_clock()
        sent = [
            Marker(now, {"type": "cue", "label": "left", "trial": 1}),
            Marker(now + 0.1, {"type": "state", "label": "rest"}),
        ]
        outlet.push_markers(sent)
        received: list[Marker] = []
        deadline = time.monotonic() + 5
        while len(received) < 2 and time.monotonic() < deadline:
            received += inlet.read_markers(timeout=0.2)
    assert [m.payload for m in received] == [m.payload for m in sent]
    assert received[0].timestamp == pytest.approx(now, abs=1e-3)


def markers_outlet(source_id: str, valid: bool = True) -> pylsl.StreamOutlet:
    """A markers stream of another program, such as the Sandbox bridge."""
    info = pylsl.StreamInfo(
        name="openmind-markers",
        type="Markers",
        channel_count=1 if valid else 2,
        nominal_srate=pylsl.IRREGULAR_RATE,
        channel_format=pylsl.cf_string,
        source_id=source_id,
    )
    info.desc().append_child_value("spec_version", "1.2.0")
    return pylsl.StreamOutlet(info)


def collect(inlet: EEGInlet, n: int, timeout: float = 5.0) -> list[Marker]:
    received: list[Marker] = []
    deadline = time.monotonic() + timeout
    while len(received) < n and time.monotonic() < deadline:
        received += inlet.read_markers(timeout=0.2)
    return received


def test_markers_of_several_producers_are_merged_in_time_order() -> None:
    source_id = unique_id()
    sandbox_id = f"openmind-sandbox:{uuid.uuid4()}:markers"
    with (
        EEGOutlet(INFO, source_id=source_id) as outlet,
        EEGInlet(
            timeout=5,
            source_id=source_id,
            markers_from=[f"{source_id}:markers", sandbox_id],
            rescan_s=0.2,
        ) as inlet,
    ):
        # The Sandbox bridge starts after the consumer: it must still be found.
        sandbox = markers_outlet(sandbox_id)
        deadline = time.monotonic() + 5
        while sandbox_id not in inlet.marker_sources and time.monotonic() < deadline:
            inlet.read_markers(timeout=0.2)
        assert sorted(inlet.marker_sources) == sorted([f"{source_id}:markers", sandbox_id])
        time.sleep(0.3)
        now = pylsl.local_clock()
        sandbox.push_sample(['{"type":"trial_start","trial":1,"source":"sandbox"}'], now)
        sandbox.push_sample(
            ['{"type":"cue","label":"left","trial":1,"source":"sandbox"}'], now + 0.2
        )
        outlet.push_markers([Marker(now + 0.1, {"type": "state", "label": "left"})])
        received = collect(inlet, 3)
    assert [m.payload["type"] for m in received] == ["trial_start", "state", "cue"]
    assert [m.timestamp for m in received] == sorted(m.timestamp for m in received)


def test_markers_from_ignores_other_producers() -> None:
    source_id = unique_id()
    other = markers_outlet(f"openmind-sandbox:{uuid.uuid4()}:markers")
    with (
        EEGOutlet(INFO, source_id=source_id),
        EEGInlet(timeout=5, source_id=source_id, markers_from=[], rescan_s=0.1) as inlet,
    ):
        time.sleep(0.5)
        other.push_sample(['{"type":"cue","label":"left"}'])
        assert collect(inlet, 1, timeout=1.0) == []
        assert not inlet.has_markers


def test_markers_stream_breaking_the_specification_is_ignored() -> None:
    source_id = unique_id()
    bad_id = f"openmind-sandbox:{uuid.uuid4()}:markers"
    bad = markers_outlet(bad_id, valid=False)
    with (
        EEGOutlet(INFO, source_id=source_id),
        EEGInlet(timeout=5, source_id=source_id, markers_from=[bad_id], rescan_s=0.1) as inlet,
    ):
        time.sleep(0.5)
        inlet.read_markers()
        assert bad_id not in inlet.marker_sources
    del bad


def test_outlet_converts_units_to_microvolts() -> None:
    info = INFO.model_copy(
        update={"channels": (ChannelInfo(label="C3", unit="mV"), ChannelInfo(label="C4"))}
    )
    source_id = unique_id()
    with (
        EEGOutlet(info, source_id=source_id) as outlet,
        EEGInlet(timeout=5, source_id=source_id) as inlet,
    ):
        assert outlet.wait_for_consumers(5)
        outlet.push(np.array([[0.05, 7.0]], dtype=np.float32), np.array([pylsl.local_clock()]))
        received, _ = read_until(inlet, 1)
    np.testing.assert_allclose(received, [[50.0, 7.0]], rtol=1e-6)


def test_closed_outlet_rejects_pushes() -> None:
    outlet = EEGOutlet(INFO, source_id=unique_id())
    outlet.close()
    outlet.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        outlet.push(np.zeros((1, 8), dtype=np.float32), np.zeros(1))


def test_stream_publishes_synthetic_source_with_markers() -> None:
    source_id = unique_id()
    source = SyntheticSource(seed=0)
    with (
        streaming(source, source_id),
        EEGInlet(timeout=5, source_id=source_id, markers_from=[f"{source_id}:markers"]) as inlet,
    ):
        time.sleep(0.3)
        source.set_state("left")
        samples, stamps = read_until(inlet, 300)
        markers: list[Marker] = []
        deadline = time.monotonic() + 3
        while not markers and time.monotonic() < deadline:
            markers += inlet.read_markers(timeout=0.2)
    assert samples.shape[1] == 8
    assert np.all(np.diff(stamps) > 0)
    assert markers[0].payload["label"] == "left"


def test_stream_stops_after_duration_and_counts() -> None:
    stats = stream(SyntheticSource(seed=0), duration=0.3, outlet=None)
    assert 0.3 <= stats.elapsed < 1.0
    assert 50 <= stats.n_samples <= 120


def test_check_stream_validates_synthetic_source() -> None:
    source_id = unique_id()
    with streaming(SyntheticSource(seed=0), source_id):
        report = check_stream(timeout=5, duration=1.5, source_id=source_id)
    assert report.ok, report.errors
    assert report.warnings == []
    assert report.info is not None
    assert report.info.labels == get_montage("A").labels
    assert report.n_samples > 200
    assert report.effective_srate == pytest.approx(250.0, rel=0.05)


def test_inlet_not_found() -> None:
    with pytest.raises(StreamNotFoundError, match="no LSL stream"):
        EEGInlet(timeout=0.3, source_id=unique_id())


def test_cli_check_stream_ok_and_not_found() -> None:
    runner = CliRunner()
    source_id = unique_id()
    with streaming(SyntheticSource(seed=0), source_id):
        result = runner.invoke(app, ["check-stream", "--source-id", source_id, "--duration", "1"])
    assert result.exit_code == 0, result.output
    assert "OK: the stream follows the specification." in result.output
    assert "8 channels @ 250 Hz" in result.output

    result = runner.invoke(app, ["check-stream", "--source-id", unique_id(), "--timeout", "0.3"])
    assert result.exit_code == 2
