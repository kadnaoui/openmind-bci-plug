"""Common interface and metadata shared by every EEG source.

Every source (synthetic, replay, LSL relay, hardware driver) describes itself with a
:class:`StreamInfo` and implements :class:`EEGSource`. The LSL output only relies on
this interface, so any new driver gets the standard stream for free.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Units accepted for a channel. The standard stream uses microvolts (``"uV"``).
KNOWN_UNITS: frozenset[str] = frozenset({"uV", "mV", "V"})

#: Channel types accepted for a channel.
KNOWN_CHANNEL_TYPES: frozenset[str] = frozenset({"EEG", "EOG", "EMG", "ECG", "MISC"})

#: Array of samples, shape ``(n_samples, n_channels)``, in the channels' units.
SampleArray = npt.NDArray[np.float32]

#: LSL timestamps in seconds (``pylsl.local_clock()``), shape ``(n_samples,)``.
TimestampArray = npt.NDArray[np.float64]


class ChannelInfo(BaseModel):
    """Description of one channel of a stream.

    Args:
        label: Electrode label, preferably from the 10-20/10-10 system (``"C3"``).
        unit: Physical unit of the samples; one of :data:`KNOWN_UNITS`.
        type: Channel type; one of :data:`KNOWN_CHANNEL_TYPES`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)
    unit: str = "uV"
    type: str = "EEG"

    @field_validator("label")
    @classmethod
    def _label_is_clean(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError(f"channel label {value!r} has leading or trailing whitespace")
        return value

    @field_validator("unit")
    @classmethod
    def _unit_is_known(cls, value: str) -> str:
        if value not in KNOWN_UNITS:
            raise ValueError(f"unknown unit {value!r}; expected one of {sorted(KNOWN_UNITS)}")
        return value

    @field_validator("type")
    @classmethod
    def _type_is_known(cls, value: str) -> str:
        if value not in KNOWN_CHANNEL_TYPES:
            raise ValueError(
                f"unknown channel type {value!r}; expected one of {sorted(KNOWN_CHANNEL_TYPES)}"
            )
        return value


class StreamInfo(BaseModel):
    """Metadata of an EEG stream, as published in the LSL ``desc`` field.

    Args:
        name: Stream name (``"openmind-eeg"`` for the standard stream).
        srate: Nominal sampling rate in Hz; must be strictly positive.
        channels: Ordered channel descriptions; labels must be unique
            (case-insensitive) and the list must not be empty.
        manufacturer: Hardware manufacturer, or ``"OpenMind"`` for software sources.
        model: Device model or source kind (``"IronBCI-8"``, ``"synthetic"``...).
        driver: Name of the plug driver that produced the stream.
        reference: Reference electrode or scheme (``"Cz"``, ``"average"``...).
        montage: Name of the montage in use (``"A"``), if any.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(default="openmind-eeg", min_length=1)
    srate: float = Field(gt=0, allow_inf_nan=False)
    channels: tuple[ChannelInfo, ...] = Field(min_length=1)
    manufacturer: str = "unknown"
    model: str = "unknown"
    driver: str = "unknown"
    reference: str = "unknown"
    montage: str | None = None

    @model_validator(mode="after")
    def _labels_are_unique(self) -> Self:
        seen: set[str] = set()
        duplicates: list[str] = []
        for label in self.labels:
            key = label.casefold()
            if key in seen:
                duplicates.append(label)
            seen.add(key)
        if duplicates:
            raise ValueError(f"duplicate channel labels: {duplicates}")
        return self

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return len(self.channels)

    @property
    def labels(self) -> tuple[str, ...]:
        """Channel labels, in stream order."""
        return tuple(channel.label for channel in self.channels)


@dataclass(frozen=True)
class Marker:
    """An event attached to the EEG stream (cue, state change, artifact...).

    Args:
        timestamp: Time of the event in seconds, on the LSL clock, aligned with the
            timestamp of the first EEG sample it applies to.
        payload: JSON-serializable content; by convention it has a ``"type"`` key
            (``{"type": "state", "label": "left"}``).
    """

    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize the payload as compact JSON, as sent on the markers stream."""
        return json.dumps(self.payload, separators=(",", ":"), sort_keys=True)


class SourceStateError(RuntimeError):
    """Raised when a source is used in the wrong state (e.g. read before start)."""


class EEGSource(ABC):
    """Base class of every EEG source.

    Subclasses implement :meth:`_start`, :meth:`_stop` and :meth:`_read`; the public
    methods handle state checks and validate what the subclass returns, so every
    source honours the same contract.

    Typical use::

        with SomeSource(...) as source:
            samples, timestamps = source.read(max_samples=64)
    """

    def __init__(self) -> None:
        self._running = False

    @property
    @abstractmethod
    def info(self) -> StreamInfo:
        """Metadata of the stream produced by this source."""

    @property
    def is_running(self) -> bool:
        """Whether the source has been started and not stopped."""
        return self._running

    def start(self) -> None:
        """Start acquisition. Calling it on a running source does nothing."""
        if self._running:
            return
        self._start()
        self._running = True

    def stop(self) -> None:
        """Stop acquisition. Calling it on a stopped source does nothing."""
        if not self._running:
            return
        try:
            self._stop()
        finally:
            self._running = False

    def read(self, max_samples: int) -> tuple[SampleArray, TimestampArray]:
        """Return the samples available now, without blocking for long.

        Args:
            max_samples: Maximum number of samples to return; must be positive.

        Returns:
            ``(samples, timestamps)``: ``samples`` has shape
            ``(n_samples, n_channels)`` with ``0 <= n_samples <= max_samples``, in the
            channels' units; ``timestamps`` has shape ``(n_samples,)``, in seconds on
            the LSL clock.

        Raises:
            ValueError: If ``max_samples`` is not positive.
            SourceStateError: If the source is not running, or the subclass returned
                arrays that break the contract.
        """
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        if not self._running:
            raise SourceStateError("source is not running; call start() first")

        samples, timestamps = self._read(max_samples)
        samples = np.asarray(samples, dtype=np.float32)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        n_channels = self.info.n_channels
        if samples.ndim != 2 or samples.shape[1] != n_channels:
            raise SourceStateError(
                f"source returned samples of shape {samples.shape}, "
                f"expected (n_samples, {n_channels})"
            )
        if samples.shape[0] > max_samples:
            raise SourceStateError(
                f"source returned {samples.shape[0]} samples, more than max_samples={max_samples}"
            )
        if timestamps.shape != (samples.shape[0],):
            raise SourceStateError(
                f"source returned timestamps of shape {timestamps.shape}, "
                f"expected ({samples.shape[0]},)"
            )
        return samples, timestamps

    def read_markers(self) -> list[Marker]:
        """Return and clear the markers produced since the last call.

        Markers are only returned once the EEG samples they refer to have been
        returned by :meth:`read`. Sources without markers return an empty list.
        """
        return []

    def impedances(self) -> dict[str, float] | None:
        """Last measured electrode impedances in kΩ, by label, or ``None``.

        Only drivers whose hardware measures impedance override this.
        """
        return None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    @abstractmethod
    def _start(self) -> None:
        """Open the device or file and begin producing samples."""

    @abstractmethod
    def _stop(self) -> None:
        """Release the device or file."""

    @abstractmethod
    def _read(self, max_samples: int) -> tuple[SampleArray, TimestampArray]:
        """Return up to ``max_samples`` samples and their LSL timestamps."""
