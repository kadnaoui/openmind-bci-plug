"""Tests for the stream specification helpers (no network needed)."""

from __future__ import annotations

import pylsl
import pytest

from openmind_plug import __version__
from openmind_plug.montage import get_montage
from openmind_plug.sources import ChannelInfo, StreamInfo
from openmind_plug.spec import (
    EEG_STREAM_NAME,
    MARKERS_STREAM_NAME,
    SPEC_VERSION,
    SpecError,
    check_eeg_info,
    check_markers_info,
    eeg_source_id,
    make_eeg_info,
    make_markers_info,
    parse_eeg_info,
)


def make_info(**overrides: object) -> StreamInfo:
    fields: dict[str, object] = {
        "srate": 250.0,
        "channels": get_montage("A").channels(),
        "manufacturer": "OpenMind",
        "model": "synthetic",
        "driver": "synthetic",
        "reference": "Cz",
        "montage": "A",
    }
    fields.update(overrides)
    return StreamInfo.model_validate(fields)


def raw_eeg_info(
    name: str = EEG_STREAM_NAME,
    stream_type: str = "EEG",
    n_channels: int = 2,
    srate: float = 250.0,
    fmt: int = pylsl.cf_float32,
) -> pylsl.StreamInfo:
    return pylsl.StreamInfo(name, stream_type, n_channels, srate, fmt, "test")


def test_header_follows_spec() -> None:
    lsl_info = make_eeg_info(make_info())
    assert lsl_info.name() == "openmind-eeg"
    assert lsl_info.type() == "EEG"
    assert lsl_info.channel_count() == 8
    assert lsl_info.nominal_srate() == 250.0
    assert lsl_info.channel_format() == pylsl.cf_float32
    assert lsl_info.source_id() == "openmind-plug:synthetic:synthetic"
    assert check_eeg_info(lsl_info) == []


def test_desc_content() -> None:
    desc = make_eeg_info(make_info()).desc()
    assert desc.child_value("spec_version") == SPEC_VERSION
    first = desc.child("channels").child("channel")
    assert first.child_value("label") == "FC3"
    assert first.child_value("unit") == "microvolts"
    assert first.child_value("type") == "EEG"
    location = first.child("location")
    assert float(location.child_value("X")) == pytest.approx(-60.2)
    assert float(location.child_value("Z")) == pytest.approx(55.5)
    acquisition = desc.child("acquisition")
    assert acquisition.child_value("driver") == "synthetic"
    assert acquisition.child_value("software") == "openmind-plug"
    assert acquisition.child_value("software_version") == __version__
    assert desc.child_value("reference") == "Cz"
    assert desc.child_value("montage") == "A"


def test_unknown_labels_have_no_location() -> None:
    info = make_info(channels=(ChannelInfo(label="Custom"),), montage=None)
    channel = make_eeg_info(info).desc().child("channels").child("channel")
    assert channel.child("location").empty()
    assert make_eeg_info(info).desc().child_value("montage") == ""


def test_parse_round_trip() -> None:
    info = make_info()
    assert parse_eeg_info(make_eeg_info(info)) == info


def test_parse_round_trip_without_montage_and_other_units() -> None:
    info = make_info(
        channels=(ChannelInfo(label="C3", unit="mV"), ChannelInfo(label="EOG", type="EOG")),
        montage=None,
    )
    parsed = parse_eeg_info(make_eeg_info(info))
    assert parsed.montage is None
    assert parsed.labels == ("C3", "EOG")
    assert [c.unit for c in parsed.channels] == ["uV", "uV"]  # the stream is in microvolts
    assert [c.type for c in parsed.channels] == ["EEG", "EOG"]


def test_custom_source_id() -> None:
    assert make_eeg_info(make_info(), source_id="abc").source_id() == "abc"
    assert eeg_source_id(make_info(driver="replay", model="x")) == "openmind-plug:replay:x"


def test_check_reports_header_problems() -> None:
    problems = check_eeg_info(
        raw_eeg_info(name="other", stream_type="ExG", srate=0.0, fmt=pylsl.cf_double64)
    )
    text = "\n".join(problems)
    for expected in (
        "name is 'other'",
        "type is 'ExG'",
        "float32",
        "nominal_srate",
        "spec_version is missing",
        "describes 0 channels",
        "acquisition/manufacturer is missing",
        "reference is missing",
    ):
        assert expected in text


def test_check_reports_channel_problems() -> None:
    lsl_info = raw_eeg_info(n_channels=3)
    desc = lsl_info.desc()
    desc.append_child_value("spec_version", "2.0.0")
    channels = desc.append_child("channels")
    for label, unit, kind in (
        ("C3", "microvolts", "EEG"),
        ("c3", "volts", "EEG"),
        ("", "microvolts", "brain"),
    ):
        node = channels.append_child("channel")
        node.append_child_value("label", label)
        node.append_child_value("unit", unit)
        node.append_child_value("type", kind)
    text = "\n".join(check_eeg_info(lsl_info))
    assert "spec_version is '2.0.0'" in text
    assert "channel c3 has unit 'volts'" in text
    assert "channel 3 has no label" in text
    assert "type 'brain'" in text
    assert "duplicate channel labels: ['C3', 'c3']" in text


def test_minor_versions_are_accepted() -> None:
    lsl_info = make_eeg_info(make_info())
    lsl_info.desc().child("spec_version").first_child().set_value("1.7.3")
    assert check_eeg_info(lsl_info) == []


def test_parse_rejects_invalid_header() -> None:
    with pytest.raises(SpecError) as error:
        parse_eeg_info(raw_eeg_info())
    assert "spec_version is missing" in error.value.problems


def test_markers_header() -> None:
    lsl_info = make_markers_info("openmind-plug:synthetic:synthetic")
    assert lsl_info.name() == MARKERS_STREAM_NAME
    assert lsl_info.source_id() == "openmind-plug:synthetic:synthetic:markers"
    assert check_markers_info(lsl_info) == []


def test_markers_check_reports_problems() -> None:
    bad = pylsl.StreamInfo("m", "Events", 2, 10.0, pylsl.cf_int32, "x")
    text = "\n".join(check_markers_info(bad))
    for expected in (
        "name is 'm'",
        "type is 'Events'",
        "channel_count is 2",
        "nominal_srate is 10.0",
        "not string",
        "spec_version is missing",
    ):
        assert expected in text
