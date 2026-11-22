"""BrainFlow source: any board supported by BrainFlow, as a standard stream.

`BrainFlow <https://brainflow.org>`_ drives many EEG boards (OpenBCI Cyton and
Ganglion, Muse, Neurosity, g.tec Unicorn...) behind one API, and ships a
**synthetic board** that needs no hardware. :class:`BrainFlowSource` wraps a
BrainFlow board as an :class:`~openmind_plug.sources.EEGSource`:

- only the board's EEG rows are read; BrainFlow returns them in microvolts;
- channel labels come from ``labels`` (the electrodes you wired, in board input
  order) or, by default, from the board's documented electrode names;
- a montage can select and reorder channels by label;
- BrainFlow timestamps (Unix time, stamped by the BrainFlow acquisition thread)
  are moved to the LSL clock.

Needs the ``brainflow`` extra: ``pip install "openmind-plug[brainflow]"``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pylsl

from openmind_plug.montage import Montage, get_montage, normalize_label
from openmind_plug.sources.base import (
    ChannelInfo,
    EEGSource,
    SampleArray,
    StreamInfo,
    TimestampArray,
)

if TYPE_CHECKING:
    from brainflow.board_shim import BoardShim

#: Install hint shown when the extra is missing.
BRAINFLOW_INSTALL: Final[str] = 'pip install "openmind-plug[brainflow]"'

#: Seconds of data BrainFlow keeps in its ring buffer between two reads.
BUFFER_SECONDS: Final[float] = 60.0


class BrainFlowSourceError(ValueError):
    """Raised when a BrainFlow board cannot be exposed as configured."""


class BrainFlowSource(EEGSource):
    """Read the EEG channels of a BrainFlow board.

    The board description is read when the source is created, so :attr:`info` is
    known immediately; the board session is opened by :meth:`start` and released
    by :meth:`stop`.

    Args:
        board_id: BrainFlow board id (``-1`` for the synthetic board) or the name
            of a ``BoardIds`` member (``"SYNTHETIC_BOARD"``, ``"CYTON_BOARD"``),
            case-insensitive.
        params: Connection parameters, as attributes of ``BrainFlowInputParams``
            (``{"serial_port": "/dev/ttyUSB0"}``, ``{"mac_address": "..."}``).
        labels: Electrode label of each EEG input of the board, in board order.
            Defaults to the electrode names documented by BrainFlow for the board
            (``Ch1``, ``Ch2``... when it has none). Give them whenever your cap
            differs from the board's default placement.
        montage: Montage name or object to select and order channels by label.
        reference: Reference electrode or scheme, for the stream metadata.

    Raises:
        ImportError: If the ``brainflow`` extra is not installed.
        BrainFlowSourceError: If the board id, a parameter, the labels or the
            montage do not match the board.
    """

    def __init__(
        self,
        board_id: int | str,
        params: Mapping[str, Any] | None = None,
        labels: Sequence[str] | None = None,
        montage: str | Montage | None = None,
        reference: str = "unknown",
    ) -> None:
        super().__init__()
        board_shim, input_params_cls = _import_brainflow()
        self._board_id = resolve_board_id(board_id)
        self._input_params = input_params_cls()
        for key, value in (params or {}).items():
            if key.startswith("_") or not hasattr(self._input_params, key):
                raise BrainFlowSourceError(
                    f"unknown BrainFlow parameter {key!r}; expected one of "
                    f"{sorted(k for k in vars(self._input_params) if not k.startswith('_'))}"
                )
            setattr(self._input_params, key, value)

        descr = board_shim.get_board_descr(self._board_id)
        eeg_rows: list[int] = list(descr.get("eeg_channels", []))
        if not eeg_rows:
            raise BrainFlowSourceError(f"board {self._board_id} has no EEG channels")
        default_names = [n for n in str(descr.get("eeg_names", "")).split(",") if n]
        board_labels = choose_labels(len(eeg_rows), labels, default_names)

        resolved = get_montage(montage) if isinstance(montage, str) else montage
        picks = pick_montage(board_labels, resolved)
        self._rows = np.array([eeg_rows[i] for i in picks], dtype=np.intp)
        self._timestamp_row = int(descr["timestamp_channel"])
        board_name = str(descr.get("name", self._board_id))
        self._info = StreamInfo(
            srate=float(descr["sampling_rate"]),
            channels=tuple(ChannelInfo(label=board_labels[i], unit="uV") for i in picks),
            manufacturer="BrainFlow" if self._board_id < 0 else "unknown",
            model=board_name,
            driver="brainflow",
            reference=reference,
            montage=resolved.name if resolved is not None else None,
        )
        self._board_shim_cls = board_shim
        self._board: BoardShim | None = None

    @property
    def info(self) -> StreamInfo:
        return self._info

    @property
    def board_id(self) -> int:
        """Numeric BrainFlow board id."""
        return self._board_id

    def _start(self) -> None:
        board = self._board_shim_cls(self._board_id, self._input_params)
        board.prepare_session()
        try:
            board.start_stream(max(1, int(BUFFER_SECONDS * self._info.srate)))
        except Exception:
            board.release_session()
            raise
        self._board = board

    def _stop(self) -> None:
        board, self._board = self._board, None
        if board is None:
            return
        try:
            board.stop_stream()
        finally:
            board.release_session()

    def _read(self, max_samples: int) -> tuple[SampleArray, TimestampArray]:
        assert self._board is not None  # guaranteed by EEGSource.read
        data = np.asarray(self._board.get_board_data(max_samples))
        if data.ndim != 2 or data.shape[1] == 0:
            return (
                np.empty((0, self._info.n_channels), dtype=np.float32),
                np.empty(0, dtype=np.float64),
            )
        samples = data[self._rows].T.astype(np.float32)
        return samples, unix_to_lsl(data[self._timestamp_row])


def resolve_board_id(board_id: int | str) -> int:
    """Return the numeric id of a board given by id or ``BoardIds`` name.

    Raises:
        BrainFlowSourceError: If no BrainFlow board has this id or name.
    """
    from brainflow.board_shim import BoardIds

    if isinstance(board_id, str):
        text = board_id.strip()
        try:
            return int(text)
        except ValueError:
            pass
        names = {member.name.casefold(): member.value for member in BoardIds}
        key = text.casefold()
        for candidate in (key, f"{key}_board"):
            if candidate in names:
                return int(names[candidate])
        raise BrainFlowSourceError(
            f"unknown BrainFlow board {board_id!r}; use a BoardIds name such as "
            "'SYNTHETIC_BOARD' or a numeric id"
        )
    if board_id not in {member.value for member in BoardIds}:
        raise BrainFlowSourceError(f"unknown BrainFlow board id {board_id}")
    return board_id


def choose_labels(
    n_inputs: int, labels: Sequence[str] | None, default_names: Sequence[str]
) -> list[str]:
    """Return one normalized, unique label per EEG input of a board.

    Args:
        n_inputs: Number of EEG inputs of the board.
        labels: Labels given by the user, in board order, or ``None``.
        default_names: Electrode names documented for the board (may be empty).

    Raises:
        BrainFlowSourceError: If ``labels`` has the wrong length or duplicates.
    """
    if labels is not None:
        if len(labels) != n_inputs:
            raise BrainFlowSourceError(
                f"the board has {n_inputs} EEG inputs, got {len(labels)} labels"
            )
        chosen = [normalize_label(label) for label in labels]
    elif len(default_names) == n_inputs:
        chosen = [normalize_label(name) for name in default_names]
    else:
        chosen = [f"Ch{i + 1}" for i in range(n_inputs)]
    duplicates = sorted({label for label in chosen if chosen.count(label) > 1})
    if duplicates:
        raise BrainFlowSourceError(f"duplicate channel labels {duplicates}")
    return chosen


def pick_montage(board_labels: Sequence[str], montage: Montage | None) -> list[int]:
    """Return the board input indices to read, in montage order.

    Raises:
        BrainFlowSourceError: If the board lacks electrodes of the montage.
    """
    if montage is None:
        return list(range(len(board_labels)))
    index = {label: i for i, label in enumerate(board_labels)}
    missing = [label for label in montage.labels if label not in index]
    if missing:
        raise BrainFlowSourceError(
            f"the board has no {missing} required by montage {montage.name!r}; "
            f"its inputs are labelled {list(board_labels)} (pass labels to name the "
            "electrodes you wired)"
        )
    return [index[label] for label in montage.labels]


def unix_to_lsl(stamps: Any) -> TimestampArray:
    """Move Unix timestamps (``time.time()``) to the LSL clock.

    The offset between the two clocks is measured at each call, so a step of the
    system clock (NTP) only affects the block being read.
    """
    offset = float(pylsl.local_clock()) - time.time()
    return np.asarray(stamps, dtype=np.float64) + offset


def _import_brainflow() -> tuple[Any, Any]:
    try:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"the BrainFlow source needs the brainflow extra: {BRAINFLOW_INSTALL}"
        ) from error
    return BoardShim, BrainFlowInputParams
