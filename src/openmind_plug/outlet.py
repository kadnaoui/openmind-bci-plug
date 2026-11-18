"""Publish an EEG source as the standard OpenMind LSL streams.

:class:`EEGOutlet` opens the EEG stream and its markers stream (see
``docs/stream-spec.md``); :func:`stream` runs the acquisition loop that reads a source
and pushes everything it produces.

Example::

    from openmind_plug.outlet import stream
    from openmind_plug.sources.synthetic import SyntheticSource

    stream(SyntheticSource(seed=0), duration=60)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Self

import numpy as np
import pylsl

from openmind_plug.sources.base import EEGSource, Marker, SampleArray, StreamInfo, TimestampArray
from openmind_plug.spec import TO_MICROVOLTS, eeg_source_id, make_eeg_info, make_markers_info


class EEGOutlet:
    """The standard EEG stream and its markers stream, for one source.

    Args:
        info: Metadata of the source.
        source_id: LSL ``source_id`` of the EEG stream; defaults to
            ``openmind-plug:<driver>:<model>``. The markers stream uses the same id
            followed by ``:markers``.
        max_buffered: Seconds of data LSL keeps for slow consumers.
    """

    def __init__(
        self, info: StreamInfo, source_id: str | None = None, max_buffered: int = 360
    ) -> None:
        self.info = info
        self.source_id = source_id or eeg_source_id(info)
        scale = np.array([TO_MICROVOLTS[c.unit] for c in info.channels], dtype=np.float32)
        self._scale = None if np.all(scale == 1.0) else scale
        self._eeg = pylsl.StreamOutlet(make_eeg_info(info, self.source_id), 0, max_buffered)
        self._markers = pylsl.StreamOutlet(make_markers_info(self.source_id), 0, max_buffered)
        self._closed = False

    def push(self, samples: SampleArray, timestamps: TimestampArray) -> None:
        """Push a block of samples, converted to microvolts, with their timestamps."""
        if self._closed:
            raise RuntimeError("outlet is closed")
        if samples.shape[0] == 0:
            return
        data = samples if self._scale is None else samples * self._scale
        self._eeg.push_chunk(np.ascontiguousarray(data, dtype=np.float32), timestamps.tolist())

    def push_markers(self, markers: list[Marker]) -> None:
        """Push markers as JSON strings, each with its own timestamp."""
        if self._closed:
            raise RuntimeError("outlet is closed")
        for marker in markers:
            self._markers.push_sample([marker.to_json()], marker.timestamp)

    def have_consumers(self) -> bool:
        """Whether at least one inlet is connected to the EEG stream."""
        return bool(self._eeg.have_consumers())

    def wait_for_consumers(self, timeout: float) -> bool:
        """Wait until an inlet connects to the EEG stream; return whether one did."""
        return bool(self._eeg.wait_for_consumers(timeout))

    def close(self) -> None:
        """Withdraw both streams from the network."""
        if not self._closed:
            self._closed = True
            del self._eeg, self._markers

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@dataclass(frozen=True)
class StreamStats:
    """Summary of a :func:`stream` run."""

    n_samples: int
    n_markers: int
    elapsed: float


def stream(
    source: EEGSource,
    duration: float | None = None,
    poll_interval: float = 0.02,
    max_block: int = 1024,
    stop_event: threading.Event | None = None,
    outlet: EEGOutlet | None = None,
) -> StreamStats:
    """Read ``source`` and publish it on LSL until stopped.

    The loop stops after ``duration`` seconds, when ``stop_event`` is set, on
    ``KeyboardInterrupt`` (Ctrl+C), or when a finite source (such as a replay without
    loop) has no more data.

    Args:
        source: Source to publish; started and stopped by this function.
        duration: Seconds to stream; ``None`` streams until stopped.
        poll_interval: Seconds between two reads of the source.
        max_block: Maximum number of samples read at once.
        stop_event: Event that ends the loop when set (for threads and front-ends).
        outlet: Outlet to use; by default one is created for the source and closed
            at the end.
    """
    own_outlet = outlet is None
    out = outlet or EEGOutlet(source.info)
    n_samples = n_markers = 0
    start = pylsl.local_clock()
    try:
        with source:
            while True:
                while True:  # drain everything that is due
                    samples, timestamps = source.read(max_block)
                    out.push(samples, timestamps)
                    n_samples += samples.shape[0]
                    if samples.shape[0] < max_block:
                        break
                markers = source.read_markers()
                out.push_markers(markers)
                n_markers += len(markers)

                if stop_event is not None and stop_event.is_set():
                    break
                if duration is not None and pylsl.local_clock() - start >= duration:
                    break
                if getattr(source, "finished", False):
                    break
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        pass
    finally:
        if own_outlet:
            out.close()
    return StreamStats(n_samples, n_markers, pylsl.local_clock() - start)
