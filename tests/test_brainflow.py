"""Tests for the BrainFlow source.

The label and montage helpers are tested without BrainFlow; the board tests use
BrainFlow's synthetic board (no hardware) and are skipped without the extra.
"""

from __future__ import annotations

import time

import numpy as np
import pylsl
import pytest

from openmind_plug.montage import get_montage
from openmind_plug.sources import EEGSource
from openmind_plug.sources.brainflow import (
    BrainFlowSourceError,
    choose_labels,
    pick_montage,
    unix_to_lsl,
)

#: Electrode names BrainFlow documents for its synthetic board.
SYNTHETIC_NAMES = [
    "Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8",
    "F5", "F7", "F3", "F1", "F2", "F4", "F6", "F8",
]  # fmt: skip
#: Montage C on the first eight inputs, the rest unused.
MONTAGE_C_WIRING = [*get_montage("C").labels, "F5", "F7", "F3", "F1", "F2", "F4", "F6", "F8"]


class TestChooseLabels:
    def test_defaults_to_board_names_normalized(self) -> None:
        assert choose_labels(3, None, ["fz", "C3", "CZ"]) == ["Fz", "C3", "Cz"]

    def test_falls_back_to_numbered_channels(self) -> None:
        assert choose_labels(2, None, []) == ["Ch1", "Ch2"]
        assert choose_labels(2, None, ["Fz"]) == ["Ch1", "Ch2"]

    def test_user_labels_win(self) -> None:
        assert choose_labels(2, ["c3", "c4"], ["Fz", "Cz"]) == ["C3", "C4"]

    def test_rejects_wrong_count(self) -> None:
        with pytest.raises(BrainFlowSourceError, match="3 EEG inputs, got 2"):
            choose_labels(3, ["C3", "C4"], [])

    def test_rejects_duplicates(self) -> None:
        with pytest.raises(BrainFlowSourceError, match="duplicate"):
            choose_labels(2, ["C3", "c3"], [])


class TestPickMontage:
    def test_all_inputs_without_montage(self) -> None:
        assert pick_montage(["C3", "Cz"], None) == [0, 1]

    def test_montage_order(self) -> None:
        labels = list(reversed(MONTAGE_C_WIRING))
        picks = pick_montage(labels, get_montage("C"))
        assert [labels[i] for i in picks] == list(get_montage("C").labels)

    def test_missing_electrodes_are_named(self) -> None:
        with pytest.raises(BrainFlowSourceError, match=r"FC3.*montage 'C'.*pass labels"):
            pick_montage(SYNTHETIC_NAMES, get_montage("C"))


def test_unix_to_lsl_maps_now_to_now() -> None:
    (converted,) = unix_to_lsl([time.time()])
    assert converted == pytest.approx(pylsl.local_clock(), abs=0.05)


@pytest.fixture
def brainflow_module() -> None:
    pytest.importorskip("brainflow", reason="needs the brainflow extra")


@pytest.mark.usefixtures("brainflow_module")
class TestSyntheticBoard:
    def make(self, **kwargs: object) -> EEGSource:
        from openmind_plug.sources.brainflow import BrainFlowSource

        return BrainFlowSource("synthetic", **kwargs)  # type: ignore[arg-type]

    def test_info_from_board_description(self) -> None:
        source = self.make()
        info = source.info
        assert info.srate == 250.0
        assert info.labels == tuple(SYNTHETIC_NAMES)
        assert {c.unit for c in info.channels} == {"uV"}
        assert (info.driver, info.model, info.manufacturer) == (
            "brainflow",
            "Synthetic",
            "BrainFlow",
        )
        assert info.montage is None

    def test_board_id_forms(self) -> None:
        from openmind_plug.sources.brainflow import resolve_board_id

        assert resolve_board_id("SYNTHETIC_BOARD") == -1
        assert resolve_board_id("synthetic") == -1
        assert resolve_board_id("-1") == -1
        assert resolve_board_id(-1) == -1
        with pytest.raises(BrainFlowSourceError, match="unknown BrainFlow board"):
            resolve_board_id("no_such_board")
        with pytest.raises(BrainFlowSourceError, match="unknown BrainFlow board id"):
            resolve_board_id(12345)

    def test_rejects_unknown_parameter(self) -> None:
        with pytest.raises(BrainFlowSourceError, match="unknown BrainFlow parameter 'port'"):
            self.make(params={"port": "/dev/ttyUSB0"})

    def test_montage_from_wired_labels(self) -> None:
        source = self.make(labels=MONTAGE_C_WIRING, montage="C", reference="A1")
        assert source.info.labels == get_montage("C").labels
        assert source.info.montage == "C"
        assert source.info.reference == "A1"

    def test_reads_microvolts_on_the_lsl_clock(self) -> None:
        source = self.make(labels=MONTAGE_C_WIRING, montage="C")
        blocks: list[np.ndarray] = []
        stamps: list[np.ndarray] = []
        with source:
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                samples, timestamps = source.read(64)
                blocks.append(samples)
                stamps.append(timestamps)
                time.sleep(0.05)
            now = pylsl.local_clock()
        assert not source.is_running

        data = np.concatenate(blocks)
        ts = np.concatenate(stamps)
        assert data.shape[1] == 8
        assert data.dtype == np.float32
        assert len(data) > 150  # about 250 Hz over 1.5 s
        assert np.all(np.isfinite(data))
        assert 1.0 < float(np.std(data)) < 1000.0  # microvolts, not volts
        assert np.all(np.diff(ts) >= 0)  # oldest first, blocks in order
        assert now - 3.0 < ts[0] < ts[-1] <= now + 0.05

    def test_stop_is_safe_twice(self) -> None:
        source = self.make()
        source.start()
        source.stop()
        source.stop()
        assert not source.is_running
