"""Tests for ChannelInfo, StreamInfo and the EEGSource contract."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from openmind_plug.sources import (
    ChannelInfo,
    EEGSource,
    Marker,
    SampleArray,
    SourceStateError,
    StreamInfo,
    TimestampArray,
)


def make_info(labels: tuple[str, ...] = ("C3", "Cz", "C4"), srate: float = 250.0) -> StreamInfo:
    return StreamInfo(srate=srate, channels=tuple(ChannelInfo(label=label) for label in labels))


# --- ChannelInfo ------------------------------------------------------------


def test_channel_defaults() -> None:
    channel = ChannelInfo(label="C3")
    assert channel.unit == "uV"
    assert channel.type == "EEG"


@pytest.mark.parametrize("unit", ["uV", "mV", "V"])
def test_channel_accepts_known_units(unit: str) -> None:
    assert ChannelInfo(label="C3", unit=unit).unit == unit


@pytest.mark.parametrize("unit", ["µV", "microvolts", "", "uv"])
def test_channel_rejects_unknown_units(unit: str) -> None:
    with pytest.raises(ValidationError, match="unknown unit"):
        ChannelInfo(label="C3", unit=unit)


def test_channel_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError, match="unknown channel type"):
        ChannelInfo(label="C3", type="eeg")


@pytest.mark.parametrize("label", ["", " C3", "C3 "])
def test_channel_rejects_bad_labels(label: str) -> None:
    with pytest.raises(ValidationError):
        ChannelInfo(label=label)


def test_channel_is_immutable() -> None:
    channel = ChannelInfo(label="C3")
    with pytest.raises(ValidationError):
        channel.label = "C4"


# --- StreamInfo -------------------------------------------------------------


def test_stream_info_defaults_and_properties() -> None:
    info = make_info()
    assert info.name == "openmind-eeg"
    assert info.n_channels == 3
    assert info.labels == ("C3", "Cz", "C4")
    assert info.montage is None


@pytest.mark.parametrize("srate", [0.0, -250.0, math.inf, math.nan])
def test_stream_info_rejects_invalid_srate(srate: float) -> None:
    with pytest.raises(ValidationError):
        make_info(srate=srate)


def test_stream_info_rejects_empty_channels() -> None:
    with pytest.raises(ValidationError):
        StreamInfo(srate=250.0, channels=())


@pytest.mark.parametrize("labels", [("C3", "C3"), ("Cz", "CZ")])
def test_stream_info_rejects_duplicate_labels(labels: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="duplicate channel labels"):
        make_info(labels=labels)


def test_stream_info_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        StreamInfo(srate=250.0, channels=(ChannelInfo(label="C3"),), colour="red")  # type: ignore[call-arg]


# --- EEGSource --------------------------------------------------------------


class FakeSource(EEGSource):
    """Minimal source returning a fixed number of zero samples per read."""

    def __init__(self, n_samples: int = 4, n_channels_out: int | None = None) -> None:
        super().__init__()
        self._info = make_info()
        self.n_samples = n_samples
        self.n_channels_out = n_channels_out or self._info.n_channels
        self.calls: list[str] = []

    @property
    def info(self) -> StreamInfo:
        return self._info

    def _start(self) -> None:
        self.calls.append("start")

    def _stop(self) -> None:
        self.calls.append("stop")

    def _read(self, max_samples: int) -> tuple[SampleArray, TimestampArray]:
        n = self.n_samples
        samples = np.zeros((n, self.n_channels_out), dtype=np.float64)
        timestamps = np.arange(n, dtype=np.float64) / self._info.srate
        return samples.astype(np.float32), timestamps


def test_source_read_returns_contract_shapes_and_dtypes() -> None:
    with FakeSource(n_samples=4) as source:
        samples, timestamps = source.read(max_samples=10)
    assert samples.shape == (4, 3)
    assert samples.dtype == np.float32
    assert timestamps.shape == (4,)
    assert timestamps.dtype == np.float64


def test_source_context_manager_starts_and_stops() -> None:
    source = FakeSource()
    with source:
        assert source.is_running
    assert not source.is_running
    assert source.calls == ["start", "stop"]


def test_source_stops_even_on_error() -> None:
    source = FakeSource()
    with pytest.raises(RuntimeError), source:
        raise RuntimeError("boom")
    assert not source.is_running
    assert source.calls == ["start", "stop"]


def test_source_start_and_stop_are_idempotent() -> None:
    source = FakeSource()
    source.start()
    source.start()
    source.stop()
    source.stop()
    assert source.calls == ["start", "stop"]


def test_source_read_before_start_fails() -> None:
    with pytest.raises(SourceStateError, match="not running"):
        FakeSource().read(max_samples=10)


@pytest.mark.parametrize("max_samples", [0, -1])
def test_source_read_rejects_non_positive_max_samples(max_samples: int) -> None:
    with FakeSource() as source, pytest.raises(ValueError, match="max_samples"):
        source.read(max_samples=max_samples)


def test_source_read_rejects_wrong_channel_count() -> None:
    with FakeSource(n_channels_out=2) as source, pytest.raises(SourceStateError, match="shape"):
        source.read(max_samples=10)


def test_source_read_rejects_too_many_samples() -> None:
    with FakeSource(n_samples=20) as source, pytest.raises(SourceStateError, match="max_samples"):
        source.read(max_samples=10)


def test_source_read_accepts_empty_block() -> None:
    with FakeSource(n_samples=0) as source:
        samples, timestamps = source.read(max_samples=10)
    assert samples.shape == (0, 3)
    assert timestamps.shape == (0,)


def test_source_has_no_markers_by_default() -> None:
    with FakeSource() as source:
        assert source.read_markers() == []


# --- Marker -----------------------------------------------------------------


def test_marker_serializes_compact_sorted_json() -> None:
    marker = Marker(timestamp=12.5, payload={"type": "cue", "label": "left", "trial": 3})
    assert marker.to_json() == '{"label":"left","trial":3,"type":"cue"}'
