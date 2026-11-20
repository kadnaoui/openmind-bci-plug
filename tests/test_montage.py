"""Tests for electrode positions and named montages."""

from __future__ import annotations

import pytest

from openmind_plug.montage import (
    DEFAULT_MONTAGE,
    MONTAGE_A,
    MONTAGE_B,
    MONTAGES,
    POSITIONS_10_10,
    Montage,
    canonical_label,
    get_montage,
    position,
)
from openmind_plug.sources import StreamInfo


def test_montage_a_is_motor_imagery_default() -> None:
    assert DEFAULT_MONTAGE == "A"
    assert get_montage() is MONTAGE_A
    assert MONTAGE_A.labels == ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "Fp1")


def test_montage_b_is_ssvep_variant() -> None:
    assert MONTAGE_B.labels == ("C3", "Cz", "C4", "CP3", "CP4", "Fp1", "O1", "O2")


@pytest.mark.parametrize("name", ["A", "a", " B "])
def test_get_montage_is_case_insensitive(name: str) -> None:
    assert get_montage(name) is MONTAGES[name.strip().upper()]


def test_montage_c_has_eight_motor_electrodes() -> None:
    assert MONTAGES["C"].labels == ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "CPz")
    assert "Fp1" not in MONTAGES["C"].labels


def test_get_montage_unknown_lists_available() -> None:
    with pytest.raises(KeyError, match=r"available montages: \['A', 'B', 'C'\]"):
        get_montage("Z")


@pytest.mark.parametrize("montage", list(MONTAGES.values()), ids=list(MONTAGES))
def test_named_montages_have_eight_channels(montage: Montage) -> None:
    assert montage.n_channels == 8


@pytest.mark.parametrize("montage", list(MONTAGES.values()), ids=list(MONTAGES))
def test_montage_channels_build_a_valid_stream(montage: Montage) -> None:
    channels = montage.channels()
    assert [c.label for c in channels] == list(montage.labels)
    assert all(c.unit == "uV" and c.type == "EEG" for c in channels)
    info = StreamInfo(srate=250.0, channels=channels, montage=montage.name)
    assert info.labels == montage.labels


def test_montage_positions_follow_input_order() -> None:
    positions = MONTAGE_A.positions()
    assert list(positions) == list(MONTAGE_A.labels)
    assert positions["C3"] == POSITIONS_10_10["C3"]


def test_montage_rejects_unknown_electrode() -> None:
    with pytest.raises(ValueError, match="unknown electrodes"):
        Montage(name="X", labels=("C3", "XX9"))


def test_montage_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Montage(name="X", labels=("C3", "C3"))


def test_montage_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no electrodes"):
        Montage(name="X", labels=())


def test_canonical_label_and_position_are_case_insensitive() -> None:
    assert canonical_label("CZ") == "Cz"
    assert canonical_label(" fp1 ") == "Fp1"
    assert position("cz") == POSITIONS_10_10["Cz"]


def test_unknown_label_raises() -> None:
    with pytest.raises(KeyError, match="unknown 10-10"):
        position("XX9")


def test_positions_are_anatomically_consistent() -> None:
    # x points to the right ear, y to the nose, z to the vertex.
    assert position("C3")[0] < 0 < position("C4")[0]
    assert position("O1")[1] < position("Cz")[1] < position("Fp1")[1]
    assert position("Cz")[2] == max(p[2] for p in POSITIONS_10_10.values())
    # Every position lies on a head-sized sphere (radius 7-13 cm).
    for label, (x, y, z) in POSITIONS_10_10.items():
        radius = (x * x + y * y + z * z) ** 0.5
        assert 0.07 < radius < 0.13, label
