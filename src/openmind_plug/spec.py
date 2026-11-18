"""OpenMind LSL stream specification: build, parse and check stream metadata.

This module is the executable form of ``docs/stream-spec.md``. Producers build their
LSL headers with :func:`make_eeg_info` and :func:`make_markers_info`; consumers turn a
received header back into a :class:`~openmind_plug.sources.StreamInfo` with
:func:`parse_eeg_info`, and :func:`check_eeg_info` lists everything that breaks the
specification.
"""

from __future__ import annotations

from typing import Final

import pylsl

from openmind_plug import __version__
from openmind_plug.montage import POSITIONS_10_10
from openmind_plug.sources.base import KNOWN_CHANNEL_TYPES, ChannelInfo, StreamInfo

#: Version of the specification implemented by this module.
SPEC_VERSION: Final = "1.0.0"

EEG_STREAM_NAME: Final = "openmind-eeg"
EEG_STREAM_TYPE: Final = "EEG"
MARKERS_STREAM_NAME: Final = "openmind-markers"
MARKERS_STREAM_TYPE: Final = "Markers"

#: Unit of every channel of the standard stream, as written in ``desc``.
STREAM_UNIT: Final = "microvolts"

#: Factor converting each :data:`~openmind_plug.sources.KNOWN_UNITS` to microvolts.
TO_MICROVOLTS: Final[dict[str, float]] = {"uV": 1.0, "mV": 1e3, "V": 1e6}


class SpecError(ValueError):
    """Raised when stream metadata does not follow the specification."""

    def __init__(self, problems: list[str]) -> None:
        details = "\n- ".join(problems)
        super().__init__(f"stream does not follow the OpenMind specification:\n- {details}")
        self.problems = problems


def eeg_source_id(info: StreamInfo) -> str:
    """Default ``source_id`` of the EEG stream: ``openmind-plug:<driver>:<model>``."""
    return f"openmind-plug:{info.driver}:{info.model}"


def make_eeg_info(info: StreamInfo, source_id: str | None = None) -> pylsl.StreamInfo:
    """Build the LSL header of the standard EEG stream.

    Channel units are declared as microvolts: the outlet converts samples from the
    source units (see :data:`TO_MICROVOLTS`). Electrode locations are added for
    10-10 labels.
    """
    lsl_info = pylsl.StreamInfo(
        name=EEG_STREAM_NAME,
        type=EEG_STREAM_TYPE,
        channel_count=info.n_channels,
        nominal_srate=info.srate,
        channel_format=pylsl.cf_float32,
        source_id=source_id or eeg_source_id(info),
    )
    desc = lsl_info.desc()
    desc.append_child_value("spec_version", SPEC_VERSION)
    channels = desc.append_child("channels")
    for channel in info.channels:
        node = channels.append_child("channel")
        node.append_child_value("label", channel.label)
        node.append_child_value("unit", STREAM_UNIT)
        node.append_child_value("type", channel.type)
        if channel.label in POSITIONS_10_10:
            location = node.append_child("location")
            for axis, value in zip("XYZ", POSITIONS_10_10[channel.label], strict=True):
                location.append_child_value(axis, f"{value * 1000:.1f}")
    acquisition = desc.append_child("acquisition")
    acquisition.append_child_value("manufacturer", info.manufacturer)
    acquisition.append_child_value("model", info.model)
    acquisition.append_child_value("driver", info.driver)
    acquisition.append_child_value("software", "openmind-plug")
    acquisition.append_child_value("software_version", __version__)
    desc.append_child_value("reference", info.reference)
    desc.append_child_value("montage", info.montage or "")
    return lsl_info


def make_markers_info(eeg_source_id: str) -> pylsl.StreamInfo:
    """Build the LSL header of the markers stream paired with an EEG stream."""
    lsl_info = pylsl.StreamInfo(
        name=MARKERS_STREAM_NAME,
        type=MARKERS_STREAM_TYPE,
        channel_count=1,
        nominal_srate=pylsl.IRREGULAR_RATE,
        channel_format=pylsl.cf_string,
        source_id=f"{eeg_source_id}:markers",
    )
    lsl_info.desc().append_child_value("spec_version", SPEC_VERSION)
    return lsl_info


def check_eeg_info(lsl_info: pylsl.StreamInfo) -> list[str]:
    """Return every way an EEG stream header breaks the specification (empty if valid)."""
    problems: list[str] = []
    if lsl_info.name() != EEG_STREAM_NAME:
        problems.append(f"name is {lsl_info.name()!r}, expected {EEG_STREAM_NAME!r}")
    if lsl_info.type() != EEG_STREAM_TYPE:
        problems.append(f"type is {lsl_info.type()!r}, expected {EEG_STREAM_TYPE!r}")
    if lsl_info.channel_format() != pylsl.cf_float32:
        problems.append("channel_format is not float32")
    if not lsl_info.nominal_srate() > 0:
        problems.append(f"nominal_srate is {lsl_info.nominal_srate()}, expected > 0")

    desc = lsl_info.desc()
    problems += _check_version(desc.child_value("spec_version"))

    labels: list[str] = []
    channel = desc.child("channels").child("channel")
    while not channel.empty():
        position = len(labels) + 1
        label = channel.child_value("label")
        labels.append(label)
        if not label.strip():
            problems.append(f"channel {position} has no label")
        unit = channel.child_value("unit")
        if unit != STREAM_UNIT:
            problems.append(
                f"channel {label or position} has unit {unit!r}, expected {STREAM_UNIT!r}"
            )
        kind = channel.child_value("type")
        if kind not in KNOWN_CHANNEL_TYPES:
            problems.append(
                f"channel {label or position} has type {kind!r}, "
                f"expected one of {sorted(KNOWN_CHANNEL_TYPES)}"
            )
        channel = channel.next_sibling("channel")
    if len(labels) != lsl_info.channel_count():
        problems.append(
            f"desc describes {len(labels)} channels but channel_count is {lsl_info.channel_count()}"
        )
    folded = [label.casefold() for label in labels if label.strip()]
    duplicates = sorted({label for label in labels if folded.count(label.casefold()) > 1})
    if duplicates:
        problems.append(f"duplicate channel labels: {duplicates}")

    acquisition = desc.child("acquisition")
    for field in ("manufacturer", "model", "driver"):
        if acquisition.child(field).empty():
            problems.append(f"acquisition/{field} is missing")
    for field in ("reference", "montage"):
        if desc.child(field).empty():
            problems.append(f"{field} is missing")
    return problems


def check_markers_info(lsl_info: pylsl.StreamInfo) -> list[str]:
    """Return every way a markers stream header breaks the specification."""
    problems: list[str] = []
    if lsl_info.name() != MARKERS_STREAM_NAME:
        problems.append(f"name is {lsl_info.name()!r}, expected {MARKERS_STREAM_NAME!r}")
    if lsl_info.type() != MARKERS_STREAM_TYPE:
        problems.append(f"type is {lsl_info.type()!r}, expected {MARKERS_STREAM_TYPE!r}")
    if lsl_info.channel_count() != 1:
        problems.append(f"channel_count is {lsl_info.channel_count()}, expected 1")
    if lsl_info.nominal_srate() != pylsl.IRREGULAR_RATE:
        problems.append(f"nominal_srate is {lsl_info.nominal_srate()}, expected 0 (irregular)")
    if lsl_info.channel_format() != pylsl.cf_string:
        problems.append("channel_format is not string")
    problems += _check_version(lsl_info.desc().child_value("spec_version"))
    return problems


def parse_eeg_info(lsl_info: pylsl.StreamInfo) -> StreamInfo:
    """Convert a standard EEG stream header into a :class:`StreamInfo`.

    Raises:
        SpecError: If the header breaks the specification.
    """
    problems = check_eeg_info(lsl_info)
    if problems:
        raise SpecError(problems)
    desc = lsl_info.desc()
    channels: list[ChannelInfo] = []
    node = desc.child("channels").child("channel")
    while not node.empty():
        channels.append(ChannelInfo(label=node.child_value("label"), type=node.child_value("type")))
        node = node.next_sibling("channel")
    acquisition = desc.child("acquisition")
    return StreamInfo(
        name=lsl_info.name(),
        srate=lsl_info.nominal_srate(),
        channels=tuple(channels),
        manufacturer=acquisition.child_value("manufacturer"),
        model=acquisition.child_value("model"),
        driver=acquisition.child_value("driver"),
        reference=desc.child_value("reference"),
        montage=desc.child_value("montage") or None,
    )


def _check_version(version: str) -> list[str]:
    if not version:
        return ["spec_version is missing"]
    major = version.split(".", 1)[0]
    expected = SPEC_VERSION.split(".", 1)[0]
    if major != expected:
        return [f"spec_version is {version!r}, only {expected}.x.y is supported"]
    return []
