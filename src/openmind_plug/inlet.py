"""Consume the standard OpenMind LSL streams.

Example::

    from openmind_plug.inlet import EEGInlet

    with EEGInlet(timeout=5) as inlet:
        print(inlet.info.labels, inlet.info.srate)
        samples, timestamps = inlet.read(timeout=1.0)
        markers = inlet.read_markers()
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self

import numpy as np
import pylsl

from openmind_plug.sources.base import Marker, SampleArray, StreamInfo, TimestampArray
from openmind_plug.spec import (
    EEG_STREAM_NAME,
    EEG_STREAM_TYPE,
    MARKERS_STREAM_NAME,
    check_eeg_info,
    check_markers_info,
    parse_eeg_info,
)

logger = logging.getLogger(__name__)


class StreamNotFoundError(TimeoutError):
    """Raised when no matching LSL stream is found in time."""


def resolve_eeg_stream(timeout: float = 5.0, source_id: str | None = None) -> pylsl.StreamInfo:
    """Find the standard EEG stream on the network.

    Args:
        timeout: Seconds to wait for the stream.
        source_id: Pick the stream with this ``source_id`` when several producers run.

    Raises:
        StreamNotFoundError: If no stream is found within ``timeout``.
    """
    predicate = f"name='{EEG_STREAM_NAME}' and type='{EEG_STREAM_TYPE}'"
    if source_id:
        predicate += f" and source_id='{source_id}'"
    found = pylsl.resolve_bypred(predicate, 1, timeout)
    if not found:
        raise StreamNotFoundError(
            f"no LSL stream matching [{predicate}] found within {timeout} s; "
            "is a producer running (e.g. `openmind-plug stream --source synthetic`)?"
        )
    if len(found) > 1:
        logger.warning("%d EEG streams found, using %s", len(found), found[0].source_id())
    return found[0]


def resolve_markers_stream(eeg_source_id: str, timeout: float = 1.0) -> pylsl.StreamInfo | None:
    """Find the markers stream paired with an EEG stream, or ``None``."""
    found = pylsl.resolve_byprop("source_id", f"{eeg_source_id}:markers", 1, timeout)
    return found[0] if found else None


class EEGInlet:
    """Read the standard EEG stream and every markers stream.

    Timestamps are converted to the local LSL clock (``proc_clocksync``), so they can
    be compared with ``pylsl.local_clock()`` and with markers from other machines.

    Markers come from **every** ``openmind-markers`` stream (specification 2.3):
    the stream paired with the EEG stream, and those of other programs, such as
    the calibration scenes of OpenMind Sandbox. Streams that appear later are
    picked up while reading, and markers are merged by timestamp.

    Args:
        timeout: Seconds to wait for the streams.
        source_id: Pick a specific EEG producer when several run.
        with_markers: Also read markers streams.
        markers_from: Accept only markers streams with these ``source_id``
            values; by default, every ``openmind-markers`` stream.
        rescan_s: Seconds between two searches for new markers streams.
        max_buflen: Seconds of data buffered by LSL if reads fall behind.

    Raises:
        StreamNotFoundError: If no EEG stream is found.
        SpecError: If the EEG stream breaks the specification.
    """

    def __init__(
        self,
        timeout: float = 5.0,
        source_id: str | None = None,
        with_markers: bool = True,
        markers_from: Sequence[str] | None = None,
        rescan_s: float = 2.0,
        max_buflen: int = 360,
    ) -> None:
        resolved = resolve_eeg_stream(timeout, source_id)
        flags = pylsl.proc_clocksync | pylsl.proc_threadsafe
        self._eeg = pylsl.StreamInlet(resolved, max_buflen, processing_flags=flags)
        self.lsl_info: pylsl.StreamInfo = self._eeg.info(timeout)
        self.info: StreamInfo = parse_eeg_info(self.lsl_info)
        self.source_id: str = self.lsl_info.source_id()
        self._eeg.open_stream(timeout)

        self._flags = flags
        self._max_buflen = max_buflen
        self._markers: dict[str, pylsl.StreamInlet] = {}  # by stream uid
        self._marker_sources: dict[str, str] = {}  # uid -> source_id
        self._ignored: set[str] = set()  # uids of streams breaking the specification
        self._accepted = None if markers_from is None else set(markers_from)
        self._rescan_s = rescan_s
        self._last_scan = -math.inf
        self._resolver: pylsl.ContinuousResolver | None = None
        if with_markers:
            self._resolver = pylsl.ContinuousResolver(prop="name", value=MARKERS_STREAM_NAME)
            # Wait a little for the paired stream, which normally starts with the EEG one.
            paired = resolve_markers_stream(self.source_id, timeout=min(timeout, 2.0))
            if paired is not None and self._accepts(paired.source_id()):
                self._connect(paired, timeout)
            self._scan()

    @property
    def has_markers(self) -> bool:
        """Whether at least one markers stream is connected."""
        return bool(self._markers)

    @property
    def marker_sources(self) -> list[str]:
        """``source_id`` of every connected markers stream."""
        return sorted(self._marker_sources.values())

    def read(
        self, max_samples: int = 1024, timeout: float = 0.0
    ) -> tuple[SampleArray, TimestampArray]:
        """Return the samples received so far (up to ``max_samples``).

        Args:
            max_samples: Maximum number of samples to return.
            timeout: Seconds to wait if no sample is available yet.

        Returns:
            ``(samples, timestamps)``: ``float32`` microvolts of shape
            ``(n_samples, n_channels)`` and ``float64`` local LSL timestamps.
        """
        data, stamps = self._eeg.pull_chunk(timeout=timeout, max_samples=max_samples)
        samples = np.asarray(data, dtype=np.float32).reshape(-1, self.info.n_channels)
        return samples, np.asarray(stamps, dtype=np.float64)

    def read_markers(self, timeout: float = 0.0) -> list[Marker]:
        """Return the markers received so far from every stream, in time order.

        New markers streams are looked for every ``rescan_s`` seconds. Invalid
        JSON samples are skipped.
        """
        if self._resolver is not None and time.monotonic() - self._last_scan >= self._rescan_s:
            self._scan()
        markers: list[Marker] = []
        for index, inlet in enumerate(list(self._markers.values())):
            data, stamps = inlet.pull_chunk(timeout=timeout if index == 0 else 0.0)
            for (text,), timestamp in zip(data, stamps, strict=True):
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning("skipping marker that is not valid JSON: %r", text)
                    continue
                if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
                    logger.warning("skipping marker without a string 'type': %r", text)
                    continue
                markers.append(Marker(float(timestamp), payload))
        return sorted(markers, key=lambda marker: marker.timestamp)

    def close(self) -> None:
        """Disconnect from the streams."""
        self._eeg.close_stream()
        for inlet in self._markers.values():
            inlet.close_stream()
        self._markers.clear()

    # -- markers streams ---------------------------------------------------------

    def _accepts(self, source_id: str) -> bool:
        return self._accepted is None or source_id in self._accepted

    def _scan(self) -> None:
        """Connect to markers streams that appeared since the last scan."""
        self._last_scan = time.monotonic()
        assert self._resolver is not None
        for info in self._resolver.results():
            uid = info.uid()
            if uid in self._markers or uid in self._ignored:
                continue
            if self._accepts(info.source_id()):
                self._connect(info, timeout=1.0)

    def _connect(self, info: pylsl.StreamInfo, timeout: float) -> None:
        inlet = pylsl.StreamInlet(info, self._max_buflen, processing_flags=self._flags)
        try:
            # Resolved headers lack the description: check the full one.
            problems = check_markers_info(inlet.info(timeout))
            if problems:
                logger.warning("ignoring markers stream %s: %s", info.source_id(), problems)
                self._ignored.add(info.uid())
                return
            inlet.open_stream(timeout)
        except TimeoutError:  # pragma: no cover - depends on the network
            logger.warning("markers stream %s did not open; retrying later", info.source_id())
            return
        self._markers[info.uid()] = inlet
        self._marker_sources[info.uid()] = info.source_id()
        logger.info("reading markers from %s", info.source_id())

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@dataclass
class StreamReport:
    """Result of :func:`check_stream`."""

    source_id: str
    info: StreamInfo | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_samples: int = 0
    effective_srate: float = 0.0
    n_markers: int = 0

    @property
    def ok(self) -> bool:
        """Whether the stream follows the specification."""
        return not self.errors


def check_stream(
    timeout: float = 5.0,
    duration: float = 2.0,
    source_id: str | None = None,
    rate_tolerance: float = 0.05,
) -> StreamReport:
    """Check a live stream against the specification.

    Checks the EEG and markers headers, then receives data for ``duration`` seconds
    and checks the effective rate, timestamp order and sample values.

    Raises:
        StreamNotFoundError: If no EEG stream is found.
    """
    resolved = resolve_eeg_stream(timeout, source_id)
    report = StreamReport(source_id=resolved.source_id())
    inlet = pylsl.StreamInlet(resolved, processing_flags=pylsl.proc_clocksync)
    markers_inlet: pylsl.StreamInlet | None = None
    try:
        full = inlet.info(timeout)
        report.errors += check_eeg_info(full)
        if report.errors:
            return report
        report.info = parse_eeg_info(full)

        markers_info = resolve_markers_stream(report.source_id, timeout=min(timeout, 2.0))
        if markers_info is None:
            report.warnings.append("no paired markers stream found")
        else:
            markers_inlet = pylsl.StreamInlet(markers_info, processing_flags=pylsl.proc_clocksync)
            full_markers = markers_inlet.info(timeout)
            report.errors += [f"markers: {p}" for p in check_markers_info(full_markers)]
            markers_inlet.open_stream(timeout)

        inlet.open_stream(timeout)
        chunks: list[np.ndarray] = []
        stamps: list[float] = []
        end = pylsl.local_clock() + duration
        while pylsl.local_clock() < end:
            data, ts = inlet.pull_chunk(timeout=0.1, max_samples=4096)
            if ts:
                chunks.append(np.asarray(data, dtype=np.float32))
                stamps.extend(ts)
            if markers_inlet is not None:
                report.n_markers += len(markers_inlet.pull_chunk(timeout=0.0)[1])
        _check_data(report, chunks, stamps, rate_tolerance)
    finally:
        inlet.close_stream()
        if markers_inlet is not None:
            markers_inlet.close_stream()
    return report


def _check_data(
    report: StreamReport,
    chunks: list[np.ndarray],
    stamps: list[float],
    rate_tolerance: float,
) -> None:
    assert report.info is not None
    report.n_samples = len(stamps)
    if report.n_samples < 2:
        report.errors.append(f"received {report.n_samples} samples; is the producer streaming?")
        return
    data = np.vstack(chunks)
    times = np.asarray(stamps)
    if not np.all(np.isfinite(data)):
        report.errors.append("some samples are NaN or infinite")
    if np.any(np.diff(times) <= 0):
        report.errors.append("timestamps are not strictly increasing")
    span = float(times[-1] - times[0])
    report.effective_srate = (report.n_samples - 1) / span if span > 0 else math.inf
    nominal = report.info.srate
    if abs(report.effective_srate - nominal) > rate_tolerance * nominal:
        report.errors.append(
            f"effective rate {report.effective_srate:.1f} Hz differs from nominal "
            f"{nominal:g} Hz by more than {rate_tolerance:.0%}"
        )
