"""LSL relay source: take a third-party LSL stream and republish it in standard form.

Many tools already publish EEG over LSL (OpenBCI GUI, BrainFlow, PiEEG Server,
vendor apps), each with its own names, units and metadata. :class:`LSLSource`
reads such a stream and exposes it as an :class:`~openmind_plug.sources.EEGSource`,
so the standard outlet republishes it as ``openmind-eeg`` in microvolts:

- channel labels come from the stream description (``channels/channel/label``),
  can be renamed, and are normalized to 10-10 spelling (``"Fc3."`` → ``FC3``);
- units come from the description (``microvolts``, ``mV``, ``volts``...) or are
  given explicitly; the outlet converts them to microvolts;
- a montage can select and reorder channels.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Final

import numpy as np
import pylsl

from openmind_plug.montage import Montage, get_montage, normalize_label
from openmind_plug.sources.base import (
    KNOWN_CHANNEL_TYPES,
    KNOWN_UNITS,
    ChannelInfo,
    EEGSource,
    SampleArray,
    StreamInfo,
    TimestampArray,
)
from openmind_plug.spec import EEG_STREAM_NAME

logger = logging.getLogger(__name__)

#: Unit spellings found in LSL descriptions, mapped to plug units.
UNIT_ALIASES: Final[dict[str, str]] = {
    "microvolts": "uV",
    "microvolt": "uV",
    "uv": "uV",
    "µv": "uV",
    "μv": "uV",
    "millivolts": "mV",
    "millivolt": "mV",
    "mv": "mV",
    "volts": "V",
    "volt": "V",
    "v": "V",
}


class LSLSourceError(ValueError):
    """Raised when a third-party stream cannot be relayed as configured."""


class LSLSource(EEGSource):
    """Relay a third-party LSL stream.

    The stream is resolved when the source is created, so :attr:`info` is known
    immediately; data flow between :meth:`start` and :meth:`stop`.

    Args:
        name: Name of the stream to relay.
        stream_type: Type of the stream to relay (``"EEG"`` by default; ``None`` for any).
        source_id: ``source_id`` of the stream, to pick one producer among several.
        timeout: Seconds to wait for the stream.
        rename: Mapping of original label to new label, applied before normalization
            (``{"EXG Channel 0": "C3"}``).
        unit: Unit of the incoming samples (``"uV"``, ``"mV"``, ``"V"``), overriding the
            description. Without it, units are read from the description, and
            microvolts are assumed when it has none.
        montage: Montage name or object to select and order channels.

    Raises:
        StreamNotFoundError: If no matching stream is found.
        LSLSourceError: If the stream is the standard stream itself, is irregular,
            or lacks channels required by the montage.
    """

    def __init__(
        self,
        name: str | None = None,
        stream_type: str | None = "EEG",
        source_id: str | None = None,
        timeout: float = 5.0,
        rename: Mapping[str, str] | None = None,
        unit: str | None = None,
        montage: str | Montage | None = None,
    ) -> None:
        super().__init__()
        if name == EEG_STREAM_NAME:
            raise LSLSourceError(
                f"{EEG_STREAM_NAME!r} is already a standard stream; consume it with EEGInlet"
            )
        if unit is not None and unit not in KNOWN_UNITS:
            raise LSLSourceError(f"unknown unit {unit!r}; expected one of {sorted(KNOWN_UNITS)}")

        resolved = _resolve(name, stream_type, source_id, timeout)
        self._inlet = pylsl.StreamInlet(
            resolved,
            max_buflen=360,
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_threadsafe,
        )
        full = self._inlet.info(timeout)
        if full.nominal_srate() <= 0:
            raise LSLSourceError(
                f"stream {full.name()!r} is irregular; only regular EEG is relayed"
            )

        channels = _describe_channels(full, rename or {}, unit)
        resolved_montage = get_montage(montage) if isinstance(montage, str) else montage
        if resolved_montage is not None:
            by_label = {channel.label: i for i, channel in enumerate(channels)}
            missing = [label for label in resolved_montage.labels if label not in by_label]
            if missing:
                raise LSLSourceError(
                    f"stream {full.name()!r} lacks channels {missing} required by montage "
                    f"{resolved_montage.name!r}; available: {[c.label for c in channels]} "
                    "(use rename to map them)"
                )
            self._picks = np.array([by_label[label] for label in resolved_montage.labels])
            channels = [channels[i] for i in self._picks]
        else:
            self._picks = np.arange(len(channels))

        desc = full.desc()
        self._info = StreamInfo(
            srate=full.nominal_srate(),
            channels=tuple(channels),
            manufacturer=desc.child("acquisition").child_value("manufacturer") or "unknown",
            model=full.name(),
            driver="lsl",
            reference=_reference(desc),
            montage=resolved_montage.name if resolved_montage else None,
        )
        self._n_input = full.channel_count()
        self.upstream_source_id: str = full.source_id()

    @property
    def info(self) -> StreamInfo:
        return self._info

    def _start(self) -> None:
        self._inlet.open_stream(timeout=5.0)

    def _stop(self) -> None:
        self._inlet.close_stream()

    def _read(self, max_samples: int) -> tuple[SampleArray, TimestampArray]:
        data, stamps = self._inlet.pull_chunk(timeout=0.0, max_samples=max_samples)
        samples = np.asarray(data, dtype=np.float32).reshape(-1, self._n_input)
        return samples[:, self._picks], np.asarray(stamps, dtype=np.float64)


def _resolve(
    name: str | None, stream_type: str | None, source_id: str | None, timeout: float
) -> pylsl.StreamInfo:
    from openmind_plug.inlet import StreamNotFoundError

    clauses = [
        f"{prop}='{value}'"
        for prop, value in (("name", name), ("type", stream_type), ("source_id", source_id))
        if value
    ]
    clauses.append(f"name!='{EEG_STREAM_NAME}'")  # never relay our own output
    predicate = " and ".join(clauses)
    found = pylsl.resolve_bypred(predicate, 1, timeout)
    if not found:
        raise StreamNotFoundError(f"no LSL stream matching [{predicate}] found within {timeout} s")
    if len(found) > 1:
        logger.warning("%d matching streams, relaying %s", len(found), found[0].name())
    return found[0]


def _describe_channels(
    full: pylsl.StreamInfo, rename: Mapping[str, str], unit: str | None
) -> list[ChannelInfo]:
    n = full.channel_count()
    raw_labels: list[str] = []
    units: list[str] = []
    types: list[str] = []
    node = full.desc().child("channels").child("channel")
    while not node.empty() and len(raw_labels) < n:
        raw_labels.append(node.child_value("label"))
        units.append(node.child_value("unit"))
        types.append(node.child_value("type"))
        node = node.next_sibling("channel")
    while len(raw_labels) < n:  # no or partial description
        raw_labels.append("")
        units.append("")
        types.append("")

    channels: list[ChannelInfo] = []
    assumed_unit = False
    for i, (raw_label, raw_unit, raw_type) in enumerate(zip(raw_labels, units, types, strict=True)):
        label = normalize_label(rename.get(raw_label, raw_label) or f"Ch{i + 1}")
        if unit is not None:
            channel_unit = unit
        elif raw_unit.strip().lower() in UNIT_ALIASES:
            channel_unit = UNIT_ALIASES[raw_unit.strip().lower()]
        else:
            channel_unit, assumed_unit = "uV", True
        kind = raw_type.strip().upper()
        channels.append(
            ChannelInfo(
                label=label,
                unit=channel_unit,
                type=kind if kind in KNOWN_CHANNEL_TYPES else "EEG" if not kind else "MISC",
            )
        )
    if assumed_unit:
        logger.warning(
            "stream %r does not declare units for every channel; assuming microvolts "
            "(pass unit= to override)",
            full.name(),
        )
    labels = [c.label for c in channels]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise LSLSourceError(f"duplicate channel labels {duplicates}; use rename to fix them")
    return channels


def _reference(desc: pylsl.XMLElement) -> str:
    """Reference from a description: ``<reference>Cz</reference>`` or XDF-style label."""
    node = desc.child("reference")
    if node.empty():
        return "unknown"
    return node.child_value("label") or node.first_child().value() or "unknown"
