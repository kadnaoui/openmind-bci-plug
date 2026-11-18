"""Replay source: play back a public dataset or a recorded file as a live stream.

The whole recording is loaded in memory, restricted to the requested channels,
converted to microvolts and resampled to the target rate. :meth:`ReplaySource.read`
then releases samples at the pace of the LSL clock (optionally faster), and the
dataset events become JSON markers such as::

    {"type": "cue", "label": "left", "trial": 12, "source": "replay"}

Public datasets are downloaded with MOABB (``pip install "openmind-plug[replay]"``)
into ``$OPENMIND_DATA_DIR/mne_data``, never into the repository.
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import mne
import numpy as np
import numpy.typing as npt
import pylsl

from openmind_plug.montage import MONTAGES, Montage, get_montage, normalize_label
from openmind_plug.paths import data_dir
from openmind_plug.sources.base import (
    ChannelInfo,
    EEGSource,
    Marker,
    SampleArray,
    StreamInfo,
    TimestampArray,
)

#: Dataset event names translated to the OpenMind command vocabulary.
DEFAULT_EVENT_LABELS: Final[dict[str, str]] = {
    "left_hand": "left",
    "right_hand": "right",
    "feet": "feet",
    "both_feet": "feet",
    "tongue": "tongue",
    "hands": "hands",
    "rest": "rest",
}


@dataclass(frozen=True)
class DatasetSpec:
    """A public dataset available for replay."""

    key: str
    moabb_class: str
    description: str
    #: Signifiers MOABB/MNE use for their ``MNE_DATASETS_<SIGN>_PATH`` settings.
    mne_signs: tuple[str, ...]
    reference: str = "unknown"


#: Datasets supported by :meth:`ReplaySource.from_dataset`, by key.
DATASETS: Final[dict[str, DatasetSpec]] = {
    "bnci2014_001": DatasetSpec(
        key="bnci2014_001",
        moabb_class="BNCI2014_001",
        description="BCI Competition IV 2a: 9 subjects, 22 EEG channels, 250 Hz, "
        "left hand / right hand / feet / tongue motor imagery.",
        mne_signs=("BNCI",),
        reference="left mastoid",
    ),
    "physionetmi": DatasetSpec(
        key="physionetmi",
        moabb_class="PhysionetMI",
        description="PhysioNet EEG Motor Movement/Imagery (EEGMMIDB): 109 subjects, "
        "64 EEG channels (10-10), 160 Hz, imagined hand and feet movements.",
        mne_signs=("EEGBCI",),
    ),
}


class MissingChannelsError(ValueError):
    """Raised when the recording lacks channels required by the montage."""


@dataclass(frozen=True)
class _Event:
    index: int  # sample index in the (resampled) recording
    label: str
    kind: str = "cue"  # marker type: "cue" for dataset events, any type for recordings


class ReplaySource(EEGSource):
    """Play back an EEG recording in real time.

    Most users call :meth:`from_dataset` or :meth:`from_file`; the constructor takes
    an already loaded :class:`mne.io.Raw`.

    Args:
        raw: Recording to replay; EEG channels are expected in volts (MNE convention).
        montage: Montage name or object to select and order channels; ``None``
            keeps every EEG channel of the recording.
        srate: Target sampling rate in Hz; the recording is resampled if needed.
        speed: Playback speed; ``1.0`` is real time, ``10.0`` ten times faster
            (timestamps follow the accelerated clock).
        loop: Restart from the beginning at the end of the recording.
        event_id: Mapping of event name to stim code, used when events come from a
            stim channel instead of annotations.
        event_labels: Translation of event names to marker labels; defaults to
            :data:`DEFAULT_EVENT_LABELS`. Unknown names are kept as they are.
        model: Value of :attr:`StreamInfo.model` (e.g. dataset and subject).
        reference: Value of :attr:`StreamInfo.reference`.
        clock: Time source used by :meth:`read`; defaults to ``pylsl.local_clock``.

    Raises:
        MissingChannelsError: If the montage needs channels the recording lacks.
        ValueError: If ``srate`` or ``speed`` is not positive, or no EEG channel exists.
    """

    def __init__(
        self,
        raw: mne.io.BaseRaw,
        montage: str | Montage | None = None,
        srate: float = 250.0,
        speed: float = 1.0,
        loop: bool = False,
        event_id: Mapping[str, int] | None = None,
        event_labels: Mapping[str, str] | None = None,
        model: str = "recording",
        reference: str = "unknown",
        clock: Callable[[], float] = pylsl.local_clock,
    ) -> None:
        super().__init__()
        if not (math.isfinite(srate) and srate > 0):
            raise ValueError(f"srate must be finite and > 0, got {srate}")
        if not (math.isfinite(speed) and speed > 0):
            raise ValueError(f"speed must be finite and > 0, got {speed}")
        self._speed = speed
        self._loop = loop
        self._clock = clock
        self._lock = threading.Lock()

        resolved = get_montage(montage) if isinstance(montage, str) else montage
        picks, labels = _select_channels(raw, resolved)
        events = _extract_events(raw, event_id, event_labels or DEFAULT_EVENT_LABELS)

        eeg = raw.copy().pick(picks)
        if not eeg.preload:
            eeg.load_data(verbose=False)
        original_srate = float(eeg.info["sfreq"])
        if not math.isclose(original_srate, srate):
            eeg.resample(srate, verbose=False)
        self._data: npt.NDArray[np.float32] = (eeg.get_data() * 1e6).T.astype(np.float32)
        scale = srate / original_srate
        n_total = self._data.shape[0]
        self._events = [
            _Event(index=min(round(e.index * scale), n_total - 1), label=e.label, kind=e.kind)
            for e in events
        ]
        # Cues are numbered from 1 as trials; other markers carry no trial number.
        self._trials: list[int | None] = []
        for event in self._events:
            self._trials.append(
                sum(t is not None for t in self._trials) + 1 if event.kind == "cue" else None
            )
        self._n_cues = sum(t is not None for t in self._trials)
        if resolved is None:  # recognise a named montage from the channel list
            resolved = next((m for m in MONTAGES.values() if m.labels == tuple(labels)), None)

        self._info = StreamInfo(
            srate=srate,
            channels=tuple(ChannelInfo(label=label) for label in labels),
            manufacturer="OpenMind",
            model=model,
            driver="replay",
            reference=reference,
            montage=resolved.name if resolved else None,
        )
        self._produced = 0
        self._t0 = 0.0
        self._pending_markers: list[Marker] = []

    # --- Constructors ---------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> ReplaySource:
        """Replay a file readable by MNE (FIF, EDF, BDF, GDF, BrainVision...).

        Files written by :class:`~openmind_plug.recorder.Recorder` restore their stream
        metadata (model, reference) and markers. Keyword arguments are passed to the
        constructor and take precedence.
        """
        raw = mne.io.read_raw(Path(path), preload=True, verbose=False)
        metadata = _recording_metadata(raw)
        kwargs.setdefault("model", metadata.get("model") or Path(path).name)
        if "reference" in metadata:
            kwargs.setdefault("reference", metadata["reference"])
        return cls(raw, **kwargs)

    @classmethod
    def from_dataset(
        cls,
        dataset: str,
        subject: int = 1,
        session: str | None = None,
        runs: list[str] | None = None,
        **kwargs: Any,
    ) -> ReplaySource:
        """Replay a public dataset, downloaded on first use.

        Args:
            dataset: Key from :data:`DATASETS` (case-insensitive, ``-`` or ``_``).
            subject: Subject number, as numbered by the dataset.
            session: Session name; defaults to the first one.
            runs: Run names to concatenate, in order; defaults to all runs.
            **kwargs: Passed to the constructor (``montage``, ``srate``, ``speed``...).

        Raises:
            KeyError: If the dataset, session or a run is unknown.
            ImportError: If the ``replay`` extra (MOABB) is not installed.
        """
        spec = get_dataset(dataset)
        try:
            import moabb.datasets
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError(
                'replaying public datasets needs MOABB: pip install "openmind-plug[replay]"'
            ) from error

        moabb_dataset = getattr(moabb.datasets, spec.moabb_class)()
        if subject not in moabb_dataset.subject_list:
            raise KeyError(
                f"subject {subject} is not in {spec.key}; "
                f"available: {moabb_dataset.subject_list[0]}..{moabb_dataset.subject_list[-1]}"
            )
        with _mne_data_paths(spec.mne_signs):
            sessions = moabb_dataset.get_data(subjects=[subject])[subject]

        session_name = session or sorted(sessions)[0]
        if session_name not in sessions:
            raise KeyError(f"unknown session {session_name!r}; available: {sorted(sessions)}")
        available_runs = sessions[session_name]
        run_names = runs or list(available_runs)
        unknown = [run for run in run_names if run not in available_runs]
        if unknown:
            raise KeyError(f"unknown runs {unknown}; available: {list(available_runs)}")
        raws = [available_runs[run] for run in run_names]
        raw = mne.concatenate_raws(raws, verbose=False) if len(raws) > 1 else raws[0]

        kwargs.setdefault("event_id", moabb_dataset.event_id)
        kwargs.setdefault("model", f"{spec.moabb_class} subject {subject} session {session_name}")
        kwargs.setdefault("reference", spec.reference)
        return cls(raw, **kwargs)

    # --- Public API -----------------------------------------------------------

    @property
    def info(self) -> StreamInfo:
        return self._info

    @property
    def n_samples(self) -> int:
        """Length of the recording, in samples at the target rate."""
        return int(self._data.shape[0])

    @property
    def duration(self) -> float:
        """Length of the recording, in seconds."""
        return self.n_samples / self._info.srate

    @property
    def n_produced(self) -> int:
        """Number of samples returned so far (keeps growing when looping)."""
        return self._produced

    @property
    def finished(self) -> bool:
        """Whether a non-looping replay has returned every sample."""
        return not self._loop and self._produced >= self.n_samples

    @property
    def event_labels(self) -> tuple[str, ...]:
        """Marker labels found in the recording, in order of first appearance."""
        return tuple(dict.fromkeys(event.label for event in self._events))

    def read_markers(self) -> list[Marker]:
        with self._lock:
            markers, self._pending_markers = self._pending_markers, []
        return markers

    # --- EEGSource hooks ------------------------------------------------------

    def _start(self) -> None:
        self._t0 = self._clock() - self._produced / self._rate

    def _stop(self) -> None:
        pass

    def _read(self, max_samples: int) -> tuple[SampleArray, TimestampArray]:
        with self._lock:
            # The small epsilon absorbs float rounding (0.4 s * 250 Hz = 99.99999...).
            due = math.floor((self._clock() - self._t0) * self._rate + 1e-6) - self._produced
            n = min(max(due, 0), max_samples)
            if not self._loop:
                n = min(n, self.n_samples - self._produced)
            first = self._produced
            indices = (first + np.arange(n)) % self.n_samples
            samples = self._data[indices]
            timestamps = self._t0 + (first + np.arange(n, dtype=np.float64)) / self._rate
            self._queue_markers(first, n)
            self._produced += n
        return samples, timestamps

    # --- Internals ------------------------------------------------------------

    @property
    def _rate(self) -> float:
        """Samples released per second of clock time."""
        return self._info.srate * self._speed

    def _queue_markers(self, first: int, n: int) -> None:
        """Queue the markers of events in the sample range ``[first, first + n)``."""
        if n == 0 or not self._events:
            return
        total = self.n_samples
        for lap in range(first // total, (first + n - 1) // total + 1):
            offset = lap * total
            for event, trial in zip(self._events, self._trials, strict=True):
                index = offset + event.index
                if first <= index < first + n:
                    payload: dict[str, Any] = {"type": event.kind, "label": event.label}
                    if trial is not None:
                        payload["trial"] = lap * self._n_cues + trial
                    payload["source"] = "replay"
                    self._pending_markers.append(Marker(self._t0 + index / self._rate, payload))


def get_dataset(name: str) -> DatasetSpec:
    """Return a supported dataset by key (case-insensitive, ``-`` or ``_``).

    Raises:
        KeyError: If the dataset is not supported; the message lists the supported ones.
    """
    key = name.strip().lower().replace("-", "_")
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; supported datasets: {sorted(DATASETS)}")
    return DATASETS[key]


def _select_channels(raw: mne.io.BaseRaw, montage: Montage | None) -> tuple[list[str], list[str]]:
    """Return ``(raw channel names to pick, stream labels)`` in stream order."""
    eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    if not eeg_names:
        raise ValueError("the recording has no EEG channel")
    by_label: dict[str, str] = {}
    for name in eeg_names:
        by_label.setdefault(normalize_label(name), name)

    if montage is None:
        labels = [normalize_label(name) for name in eeg_names]
        if len(set(labels)) != len(labels):  # keep raw names if normalizing creates clashes
            labels = list(eeg_names)
        return list(eeg_names), labels

    missing = [label for label in montage.labels if label not in by_label]
    if missing:
        usable = [
            name
            for name, other in MONTAGES.items()
            if other is not montage and all(label in by_label for label in other.labels)
        ]
        hint = (
            f"montages available with this recording: {usable}"
            if usable
            else f"no named montage fits; keep all {len(eeg_names)} EEG channels with "
            "montage=None (in the CLI: omit --montage)"
        )
        raise MissingChannelsError(
            f"the recording lacks channels {missing} required by montage {montage.name!r}; " + hint
        )
    return [by_label[label] for label in montage.labels], list(montage.labels)


def _extract_events(
    raw: mne.io.BaseRaw,
    event_id: Mapping[str, int] | None,
    labels: Mapping[str, str],
) -> list[_Event]:
    """Events of the recording, at the original rate, relative to its first sample."""
    named: list[tuple[int, str]] = []
    descriptions = [
        str(d)
        for d in raw.annotations.description
        if not str(d).upper().startswith(("BAD", "EDGE"))
    ]
    if descriptions:
        events, mapping = mne.events_from_annotations(
            raw,
            event_id={d: i + 1 for i, d in enumerate(dict.fromkeys(descriptions))},
            verbose=False,
        )
        names = {code: name for name, code in mapping.items()}
        named = [(int(sample), names[int(code)]) for sample, _, code in events]
    elif event_id:
        stim = mne.pick_types(raw.info, stim=True, exclude=[])
        if len(stim):
            events = mne.find_events(
                raw, stim_channel=raw.ch_names[stim[0]], shortest_event=1, verbose=False
            )
            names = {code: name for name, code in event_id.items()}
            named = [
                (int(sample), names[int(code)]) for sample, _, code in events if int(code) in names
            ]
    result: list[_Event] = []
    for sample, name in sorted(named, key=lambda item: item[0]):  # stable: keeps file order
        kind, _, label = name.partition("/") if "/" in name else ("cue", "", name)
        if kind == "cue":
            label = labels.get(label, label)
        result.append(_Event(index=sample - raw.first_samp, label=label, kind=kind))
    return result


def _recording_metadata(raw: mne.io.BaseRaw) -> dict[str, Any]:
    """Stream metadata saved by the recorder in the file description, if any."""
    try:
        content = json.loads(raw.info.get("description") or "")
    except (TypeError, ValueError):
        return {}
    metadata = content.get("openmind") if isinstance(content, dict) else None
    return metadata if isinstance(metadata, dict) else {}


@contextmanager
def _mne_data_paths(signs: tuple[str, ...]) -> Iterator[Path]:
    """Point MNE/MOABB downloads to ``$OPENMIND_DATA_DIR/mne_data`` for a while.

    Setting the ``MNE_DATASETS_<SIGN>_PATH`` environment variables also stops MOABB
    from writing these paths into the user's global MNE configuration.
    """
    path = data_dir(create=True) / "mne_data"
    path.mkdir(parents=True, exist_ok=True)
    keys = ["MNE_DATA", *(f"MNE_DATASETS_{sign}_PATH" for sign in signs)]
    saved = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.setdefault(key, str(path))
    try:
        yield path
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
