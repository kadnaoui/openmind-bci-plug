"""Tests for the recorder, the record command and record → replay round trips."""

from __future__ import annotations

import datetime as dt
import json
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mne
import numpy as np
import pytest
from typer.testing import CliRunner

from openmind_plug.cli import app
from openmind_plug.outlet import EEGOutlet, stream
from openmind_plug.paths import DATA_DIR_ENV
from openmind_plug.recorder import (
    GAP_ANNOTATION,
    Recorder,
    default_path,
    record_source,
    validate_subject,
)
from openmind_plug.sources import ChannelInfo, Marker, StreamInfo
from openmind_plug.sources.replay import ReplaySource
from openmind_plug.sources.synthetic import SyntheticConfig, SyntheticSource
from openmind_plug.spec import SPEC_VERSION

FS = 250.0
runner = CliRunner()


class FakeClock:
    def __init__(self, now: float = 500.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def small_info(**overrides: object) -> StreamInfo:
    fields: dict[str, object] = {
        "srate": FS,
        "channels": (ChannelInfo(label="C3"), ChannelInfo(label="C4")),
        "model": "test",
        "driver": "test",
    }
    fields.update(overrides)
    return StreamInfo.model_validate(fields)


# --- Subject and paths ----------------------------------------------------------


@pytest.mark.parametrize("subject", ["P01", "P99", "P1234"])
def test_valid_subjects(subject: str) -> None:
    assert validate_subject(subject) == subject


@pytest.mark.parametrize("subject", ["p01", "P1", "Jean", "P01 Dupont", "", "S01"])
def test_rejects_non_pseudonyms(subject: str) -> None:
    with pytest.raises(ValueError, match="never a name"):
        validate_subject(subject)


def test_default_path_is_in_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    path = default_path("P03", dt.datetime(2026, 11, 18, 14, 5, 9))
    assert path == tmp_path / "recordings" / "P03" / "P03_20261118-140509_raw.fif"


# --- Recorder -----------------------------------------------------------------------


def test_add_validates_shapes() -> None:
    recorder = Recorder(small_info(), "P01")
    with pytest.raises(ValueError, match="shape"):
        recorder.add(np.zeros((4, 3), dtype=np.float32), np.zeros(4))
    with pytest.raises(ValueError, match="timestamps"):
        recorder.add(np.zeros((4, 2), dtype=np.float32), np.zeros(3))
    recorder.add(np.zeros((0, 2), dtype=np.float32), np.zeros(0))
    assert recorder.n_samples == 0


def test_empty_recording_cannot_be_saved(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing was recorded"):
        Recorder(small_info(), "P01").save(tmp_path / "x_raw.fif")


def test_save_checks_name_and_existing_file(tmp_path: Path) -> None:
    recorder = Recorder(small_info(), "P01")
    recorder.add(np.ones((10, 2), dtype=np.float32), 100 + np.arange(10) / FS)
    with pytest.raises(ValueError, match=r"_raw\.fif"):
        recorder.save(tmp_path / "rec.fif")
    path = recorder.save(tmp_path / "deep" / "rec_raw.fif")
    assert path.is_file()
    with pytest.raises(FileExistsError):
        recorder.save(path)
    assert recorder.save(path, overwrite=True) == path


def test_file_metadata(tmp_path: Path) -> None:
    info = small_info(
        channels=(ChannelInfo(label="C3"), ChannelInfo(label="EOG", type="EOG")),
        reference="Cz",
        montage=None,
    )
    recorder = Recorder(info, "P07")
    recorder.add(np.full((5, 2), 20.0, dtype=np.float32), 10 + np.arange(5) / FS)
    raw = mne.io.read_raw_fif(recorder.save(tmp_path / "meta_raw.fif"), verbose=False)

    assert raw.info["subject_info"]["his_id"] == "P07"
    assert raw.get_channel_types() == ["eeg", "eog"]
    assert raw.info["sfreq"] == FS
    metadata = json.loads(raw.info["description"])["openmind"]
    assert metadata["reference"] == "Cz"
    assert metadata["model"] == "test"
    assert metadata["spec_version"] == SPEC_VERSION
    assert "channels" not in metadata
    np.testing.assert_allclose(raw.get_data()[0], 20e-6)  # stored in volts
    positions = raw.get_montage().get_positions()["ch_pos"]  # type: ignore[union-attr]
    assert set(positions) == {"C3"}  # 10-10 positions only for known labels


def test_units_are_converted_to_microvolts(tmp_path: Path) -> None:
    info = small_info(channels=(ChannelInfo(label="C3", unit="mV"), ChannelInfo(label="C4")))
    recorder = Recorder(info, "P01")
    recorder.add(np.array([[0.02, 3.0]] * 4, dtype=np.float32), np.arange(4) / FS)
    raw = mne.io.read_raw_fif(recorder.save(tmp_path / "units_raw.fif"), verbose=False)
    np.testing.assert_allclose(raw.get_data()[:, 0], [20e-6, 3e-6], rtol=1e-6)


def test_markers_and_gaps_become_annotations(tmp_path: Path) -> None:
    recorder = Recorder(small_info(), "P01")
    recorder.add(np.zeros((100, 2), dtype=np.float32), 100 + np.arange(100) / FS)
    # 20 samples are missing after sample 99.
    recorder.add(np.zeros((50, 2), dtype=np.float32), 100 + np.arange(120, 170) / FS)
    recorder.add_markers(
        [
            Marker(100 + 10 / FS, {"type": "state", "label": "left"}),
            Marker(100 + 130 / FS, {"type": "cue", "label": "right", "trial": 1}),
            Marker(100 + 40 / FS, {"type": "custom"}),
            Marker(200.0, {"type": "cue", "label": "late"}),  # after the last sample
        ]
    )
    raw = recorder.to_raw()
    annotations = list(zip(raw.annotations.onset * FS, raw.annotations.description, strict=True))
    assert [(round(o), d) for o, d in annotations] == [
        (10, "state/left"),
        (40, "custom"),
        (100, GAP_ANNOTATION),
        (110, "cue/right"),
    ]


def test_record_source_in_real_time() -> None:
    recorder = record_source(SyntheticSource(seed=0), duration=0.3, subject="P01")
    assert recorder.n_samples == 75
    assert recorder.duration == pytest.approx(0.3)


# --- Acceptance: record 30 s of synthetic EEG, then replay it identically -----------


def test_record_then_replay_is_identical(tmp_path: Path) -> None:
    clock = FakeClock()
    source = SyntheticSource(seed=4, clock=clock, config=SyntheticConfig(line_noise_uv=3))
    recorder = Recorder(source.info, "P02")
    plan = {
        50: ("state", "left"),
        120: ("eyes", "closed"),
        170: ("blink", ""),
        220: ("state", "rest"),
    }
    with source:
        for step in range(300):  # 300 x 0.1 s = 30 s
            if step in plan:
                kind, value = plan[step]
                if kind == "state":
                    source.set_state(value)
                elif kind == "eyes":
                    source.set_eyes(value)
                else:
                    source.blink()
            clock.now += 0.1
            recorder.add(*source.read(1000))
            recorder.add_markers(source.read_markers())
    assert recorder.n_samples == 7500
    recorded = np.vstack(recorder._blocks).astype(np.float32)
    t0 = recorder._timestamps[0][0]
    sent = [
        (round((m.timestamp - t0) * FS), m.payload["type"], m.payload.get("label"))
        for m in recorder._markers
    ]
    path = recorder.save(tmp_path / "P02_synthetic_raw.fif")

    replay_clock = FakeClock(now=0.0)
    replay = ReplaySource.from_file(path, srate=FS, clock=replay_clock)
    assert replay.info.labels == source.info.labels
    assert replay.info.montage == "A"
    assert replay.info.model == "synthetic"
    assert replay.info.reference == "none"
    with replay:
        replay_clock.now += 31.0
        samples, _ = replay.read(10_000)
        markers = replay.read_markers()

    np.testing.assert_array_equal(samples, recorded)
    received = [
        (round(m.timestamp * FS), m.payload["type"], m.payload.get("label")) for m in markers
    ]
    assert received == sent
    assert [m[1] for m in sent] == ["state", "eyes", "artifact", "state"]
    assert all(m.payload["source"] == "replay" and "trial" not in m.payload for m in markers)


# --- Command line -----------------------------------------------------------------


@contextmanager
def streaming(source_id: str) -> Iterator[None]:
    stop = threading.Event()
    source = SyntheticSource(seed=0)
    outlet = EEGOutlet(source.info, source_id=source_id)
    thread = threading.Thread(
        target=stream, args=(source,), kwargs={"stop_event": stop, "outlet": outlet}
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)
        outlet.close()


def test_cli_record_saves_a_replayable_file(tmp_path: Path) -> None:
    source_id = f"openmind-plug-test:{uuid.uuid4()}"
    out = tmp_path / "P05_test_raw.fif"
    with streaming(source_id):
        result = runner.invoke(
            app,
            [
                "record",
                "--subject",
                "P05",
                "--out",
                str(out),
                "--duration",
                "1",
                "--source-id",
                source_id,
            ],
        )
    assert result.exit_code == 0, result.output
    assert "Saved" in result.output
    replay = ReplaySource.from_file(out)
    assert replay.info.montage == "A"
    assert 150 < replay.n_samples < 400


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--subject", "Alice"], "never a name"),
        (["--subject", "P01", "--out", "rec.fif"], "_raw.fif"),
        (["--subject", "P01", "--source-id", "nope", "--timeout", "0.3"], "no LSL stream"),
    ],
)
def test_cli_record_errors(
    args: list[str], message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    result = runner.invoke(app, ["record", *args])
    assert result.exit_code == 2
    assert message in result.output


def test_cli_record_refuses_existing_file_and_warns_in_git_repo(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    existing = tmp_path / "old_raw.fif"
    existing.write_bytes(b"")
    result = runner.invoke(app, ["record", "--subject", "P01", "--out", str(existing)])
    assert result.exit_code == 2
    assert "already exists" in result.output

    result = runner.invoke(
        app,
        [
            "record",
            "--subject",
            "P01",
            "--out",
            str(tmp_path / "new_raw.fif"),
            "--source-id",
            "nope",
            "--timeout",
            "0.3",
        ],
    )
    assert "inside a Git repository" in result.output
