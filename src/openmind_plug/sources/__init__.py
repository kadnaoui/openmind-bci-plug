"""EEG sources and their shared interface."""

from openmind_plug.sources.base import (
    KNOWN_CHANNEL_TYPES,
    KNOWN_UNITS,
    ChannelInfo,
    EEGSource,
    SampleArray,
    SourceStateError,
    StreamInfo,
    TimestampArray,
)

__all__ = [
    "KNOWN_CHANNEL_TYPES",
    "KNOWN_UNITS",
    "ChannelInfo",
    "EEGSource",
    "SampleArray",
    "SourceStateError",
    "StreamInfo",
    "TimestampArray",
]
