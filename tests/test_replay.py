"""Tests for the replay source.

Fast tests use a small recording generated on the fly; the ``slow`` test downloads
a public dataset into ``$OPENMIND_DATA_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

import mne
import numpy as np
import pytest

from openmind_plug.paths import DATA_DIR_ENV, data_dir
from openmind_plug.sources import EEGSource
from openmind_plug.sources.replay import (
    DATASETS,
    MissingChannelsError,
    ReplaySource,
    get_dataset,
)

SFREQ = 200.0
DURATION_S = 10.0
CHANNELS = ["Fp1", "FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "O1", "EOG1"]


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_raw(
    names: list[str] | None = None,
    sfreq: float = SFREQ,
    with_stim: bool = False,
) -> mne.io.RawArray:
    """A recording where channel ``i`` holds ``i`` microvolts plus a ramp."""
    names = names or CHANNELS
    n = int(DURATION_S * sfreq)
    types = ["eog" if name.startswith("EOG") else "eeg" for name in names]
    data = np.array([np.full(n, float(i)) for i in range(len(names))]) * 1e-6
    if with_stim:
        names, types = [*names, "STI"], [*types, "stim"]
        stim = np.zeros(n)
        stim[int(1 * sfreq)] = 1
        stim[int(4 * sfreq)] = 2
        data = np.vstack([data, stim])
    raw = mne.io.RawArray(data, mne.create_info(names, sfreq, types), verbose=False)
    if not with_stim:
        raw.set_annotations(
            mne.Annotations(
                onset=[1.0, 4.0, 5.0, 7.0],
                duration=[3.0, 3.0, 0.0, 2.0],
                description=["left_hand", "right_hand", "BAD_segment", "feet"],
            )
        )
    return raw


# --- Channels and metadata ------------------------------------------------------


def test_is_an_eeg_source() -> None:
    assert isinstance(ReplaySource(make_raw()), EEGSource)


def test_montage_selects_and_orders_channels() -> None:
    source = ReplaySource(make_raw(), montage="A", srate=SFREQ)
    assert source.info.labels == ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "Fp1")
    assert source.info.montage == "A"
    # Values encode the original channel index: FC3 was 1, Fp1 was 0.
    assert source._data[0, 0] == pytest.approx(1.0)
    assert source._data[0, -1] == pytest.approx(0.0)


def test_no_montage_keeps_all_eeg_channels() -> None:
    source = ReplaySource(make_raw(), montage=None, srate=SFREQ)
    assert source.info.labels == tuple(CHANNELS[:-1])  # EOG dropped
    assert source.info.montage is None


def test_channel_names_are_normalized() -> None:
    raw = make_raw(names=["Fc3.", "EEG CZ", "c4..", "Custom"])
    source = ReplaySource(raw, srate=SFREQ)
    assert source.info.labels == ("FC3", "Cz", "C4", "Custom")


def test_missing_channel_suggests_other_montage() -> None:
    raw = make_raw(names=["C3", "Cz", "C4", "CP3", "CP4", "Fp1", "O1", "O2"])
    with pytest.raises(MissingChannelsError, match=r"\['FC3', 'FC4'\].*available.*\['B'\]"):
        ReplaySource(raw, montage="A")


def test_missing_channel_without_alternative_suggests_none() -> None:
    raw = make_raw(names=["C3", "Cz", "C4"])
    with pytest.raises(MissingChannelsError, match="montage=None"):
        ReplaySource(raw, montage="A")


def test_converts_volts_to_microvolts() -> None:
    source = ReplaySource(make_raw(), srate=SFREQ)
    assert source._data.dtype == np.float32
    np.testing.assert_allclose(source._data[0], np.arange(9), atol=1e-4)
    assert all(c.unit == "uV" for c in source.info.channels)


def test_resamples_to_target_rate() -> None:
    source = ReplaySource(make_raw(), srate=250.0)
    assert source.info.srate == 250.0
    assert source.n_samples == int(DURATION_S * 250)
    assert source.duration == pytest.approx(DURATION_S)
    np.testing.assert_allclose(source._data[100], np.arange(9), atol=1e-3)


@pytest.mark.parametrize(("srate", "speed"), [(0, 1), (250, 0), (250, -2)])
def test_rejects_invalid_rates(srate: float, speed: float) -> None:
    with pytest.raises(ValueError):
        ReplaySource(make_raw(), srate=srate, speed=speed)


def test_rejects_recording_without_eeg() -> None:
    with pytest.raises(ValueError, match="no EEG"):
        ReplaySource(make_raw(names=["EOG1"]))


# --- Events ---------------------------------------------------------------------


def test_annotations_become_translated_events() -> None:
    source = ReplaySource(make_raw(), srate=250.0)
    assert source.event_labels == ("left", "right", "feet")  # BAD_* ignored
    assert [e.index for e in source._events] == [250, 1000, 1750]


def test_stim_channel_events_use_event_id() -> None:
    raw = make_raw(with_stim=True)
    source = ReplaySource(raw, srate=SFREQ, event_id={"left_hand": 1, "tongue": 2})
    assert source.event_labels == ("left", "tongue")
    assert [e.index for e in source._events] == [200, 800]


def test_custom_event_labels() -> None:
    source = ReplaySource(make_raw(), event_labels={"left_hand": "L"})
    assert source.event_labels == ("L", "right_hand", "feet")


# --- Real-time playback ---------------------------------------------------------


def test_read_is_paced_by_the_clock() -> None:
    clock = FakeClock()
    with ReplaySource(make_raw(), srate=SFREQ, clock=clock) as source:
        assert source.read(1000)[0].shape[0] == 0
        clock.now += 0.5
        samples, timestamps = source.read(1000)
    assert samples.shape == (100, 9)
    np.testing.assert_allclose(timestamps, 1000.0 + np.arange(100) / SFREQ)


def test_markers_are_emitted_with_sample_timestamps() -> None:
    clock = FakeClock()
    with ReplaySource(make_raw(), srate=SFREQ, clock=clock) as source:
        clock.now += 2.0
        source.read(10_000)
        markers = source.read_markers()
        assert source.read_markers() == []
    assert len(markers) == 1
    assert markers[0].timestamp == pytest.approx(1001.0)
    assert markers[0].payload == {"type": "cue", "label": "left", "trial": 1, "source": "replay"}


def test_speed_accelerates_playback() -> None:
    clock = FakeClock()
    with ReplaySource(make_raw(), srate=SFREQ, speed=10.0, clock=clock) as source:
        clock.now += 0.5
        samples, timestamps = source.read(10_000)
        markers = source.read_markers()
    assert samples.shape[0] == 1000  # 5 s of recording in 0.5 s
    assert timestamps[-1] == pytest.approx(1000.0 + 999 / (SFREQ * 10))
    assert [m.payload["label"] for m in markers] == ["left", "right"]
    assert markers[1].timestamp == pytest.approx(1000.4)


def test_stops_at_the_end_without_loop() -> None:
    clock = FakeClock()
    with ReplaySource(make_raw(), srate=SFREQ, clock=clock) as source:
        clock.now += DURATION_S + 5
        samples, _ = source.read(100_000)
        assert samples.shape[0] == source.n_samples
        assert source.finished
        assert source.read(100)[0].shape[0] == 0


def test_loop_restarts_and_numbers_trials() -> None:
    clock = FakeClock()
    with ReplaySource(make_raw(), srate=SFREQ, loop=True, clock=clock) as source:
        clock.now += DURATION_S + 2
        samples, timestamps = source.read(100_000)
        markers = source.read_markers()
        assert not source.finished
    n = source.n_samples
    assert samples.shape[0] == n + 400
    np.testing.assert_array_equal(samples[n : n + 10], samples[:10])
    assert np.all(np.diff(timestamps) > 0)
    assert [m.payload["trial"] for m in markers] == [1, 2, 3, 4]
    assert markers[3].timestamp == pytest.approx(1000.0 + DURATION_S + 1.0)


def test_restart_continues_the_timeline() -> None:
    clock = FakeClock()
    source = ReplaySource(make_raw(), srate=SFREQ, clock=clock)
    with source:
        clock.now += 1.0
        source.read(10_000)
    clock.now += 30
    with source:
        clock.now += 1.0
        samples, _ = source.read(10_000)
    assert samples.shape[0] == 200
    assert source.n_produced == 400


# --- Files and datasets -------------------------------------------------------------


def test_from_file_reads_fif(tmp_path: Path) -> None:
    path = tmp_path / "rec_raw.fif"
    make_raw().save(path, verbose=False)
    source = ReplaySource.from_file(path, montage="A", srate=SFREQ)
    assert source.info.model == "rec_raw.fif"
    assert source.info.labels[0] == "FC3"
    assert source.event_labels == ("left", "right", "feet")


@pytest.mark.parametrize("name", ["bnci2014_001", "BNCI2014-001", " PhysionetMI "])
def test_get_dataset_accepts_spellings(name: str) -> None:
    assert get_dataset(name).key in DATASETS


def test_get_dataset_unknown_lists_supported() -> None:
    with pytest.raises(KeyError, match="supported datasets"):
        get_dataset("nope")


def test_data_dir_follows_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "data"))
    assert data_dir() == tmp_path / "data"
    assert not (tmp_path / "data").exists()
    assert data_dir(create=True).is_dir()
    monkeypatch.delenv(DATA_DIR_ENV)
    assert data_dir() == Path.home() / "openmind-data"


@pytest.mark.slow
def test_replay_bnci2014_001_subject_1() -> None:
    pytest.importorskip("moabb")
    before = {key: os.environ.get(key) for key in ("MNE_DATA", "MNE_DATASETS_BNCI_PATH")}
    clock = FakeClock()
    source = ReplaySource.from_dataset(
        "bnci2014_001", subject=1, runs=["0"], speed=1.0, clock=clock
    )
    assert {key: os.environ.get(key) for key in before} == before  # environment restored
    assert (data_dir() / "mne_data").is_dir()
    assert source.info.srate == 250.0
    assert source.info.n_channels == 22
    assert source.info.labels[:3] == ("Fz", "FC3", "FC1")
    assert source.info.reference == "left mastoid"

    with source:
        clock.now += 10.0
        samples, timestamps = source.read(10_000)
        markers = source.read_markers()
    assert samples.shape == (2500, 22)
    assert np.isclose(1 / np.mean(np.diff(timestamps)), 250.0)
    assert 1 < np.median(samples.std(axis=0)) < 100  # microvolts, not volts
    assert markers, "the first 10 s of a run contain at least one cue"
    assert {m.payload["label"] for m in markers} <= {"left", "right", "feet", "tongue"}
    assert all(m.payload["type"] == "cue" for m in markers)
