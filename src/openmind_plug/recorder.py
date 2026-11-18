"""Record EEG and markers to a FIF file.

A recording keeps everything needed to replay it with
:meth:`~openmind_plug.sources.replay.ReplaySource.from_file`:

- samples, stored in volts (MNE convention) in double precision, so microvolt
  values come back unchanged;
- channel labels, types and 10-10 electrode positions;
- markers as annotations described ``"<type>/<label>"`` (``"state/left"``,
  ``"cue/right"``), at the timestamp of their EEG sample;
- gaps in the received stream as ``BAD_gap`` annotations;
- the stream metadata (:class:`~openmind_plug.sources.StreamInfo`) as JSON in the
  file description.

EEG recordings are health data: the subject is identified only by a pseudonym such as
``P01``, and files go to ``$OPENMIND_DATA_DIR/recordings/`` by default, outside any
repository.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import threading
from pathlib import Path
from typing import Any, Final

import mne
import numpy as np
import numpy.typing as npt

from openmind_plug import __version__
from openmind_plug.montage import POSITIONS_10_10
from openmind_plug.paths import data_dir
from openmind_plug.sources.base import EEGSource, Marker, SampleArray, StreamInfo, TimestampArray
from openmind_plug.spec import SPEC_VERSION, TO_MICROVOLTS

#: Pseudonymous subject identifiers: ``P`` followed by at least two digits.
SUBJECT_PATTERN: Final = re.compile(r"^P\d{2,}$")

#: Key under which the stream metadata is stored in the FIF description.
METADATA_KEY: Final = "openmind"

#: Annotation marking a hole in the received stream.
GAP_ANNOTATION: Final = "BAD_gap"

_MNE_TYPES: Final = {"EEG": "eeg", "EOG": "eog", "EMG": "emg", "ECG": "ecg", "MISC": "misc"}


def validate_subject(subject: str) -> str:
    """Return ``subject`` if it is a pseudonym like ``P01``.

    Raises:
        ValueError: Otherwise. Real names must never be stored with EEG data.
    """
    if not SUBJECT_PATTERN.fullmatch(subject):
        raise ValueError(
            f"subject {subject!r} is not a pseudonym: use P followed by digits (P01, P02...), "
            "never a name"
        )
    return subject


def default_path(subject: str, start: dt.datetime | None = None) -> Path:
    """``$OPENMIND_DATA_DIR/recordings/<subject>/<subject>_<YYYYmmdd-HHMMSS>_raw.fif``."""
    stamp = (start or dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
    return data_dir() / "recordings" / subject / f"{subject}_{stamp}_raw.fif"


class Recorder:
    """Accumulate EEG blocks and markers, then save them as a FIF file.

    Feed it from a source (:func:`record_source`), from an LSL inlet
    (:func:`record_stream`) or directly with :meth:`add` and :meth:`add_markers`.

    Args:
        info: Metadata of the recorded stream; samples are in the channels' units.
        subject: Pseudonymous subject identifier (``P01``).

    Raises:
        ValueError: If ``subject`` is not a pseudonym.
    """

    def __init__(self, info: StreamInfo, subject: str) -> None:
        self.info = info
        self.subject = validate_subject(subject)
        self.started_at = dt.datetime.now(dt.UTC)
        self._scale = np.array([TO_MICROVOLTS[c.unit] for c in info.channels])
        self._blocks: list[npt.NDArray[np.float64]] = []
        self._timestamps: list[npt.NDArray[np.float64]] = []
        self._markers: list[Marker] = []
        self._lock = threading.Lock()

    @property
    def n_samples(self) -> int:
        """Number of samples recorded so far."""
        return sum(len(t) for t in self._timestamps)

    @property
    def duration(self) -> float:
        """Recorded duration in seconds, at the nominal rate."""
        return self.n_samples / self.info.srate

    @property
    def n_markers(self) -> int:
        """Number of markers recorded so far."""
        return len(self._markers)

    def add(self, samples: SampleArray, timestamps: TimestampArray) -> None:
        """Append a block of samples (``(n_samples, n_channels)``) and their timestamps."""
        if samples.shape[0] == 0:
            return
        if samples.ndim != 2 or samples.shape[1] != self.info.n_channels:
            raise ValueError(
                f"expected samples of shape (n, {self.info.n_channels}), got {samples.shape}"
            )
        if timestamps.shape != (samples.shape[0],):
            raise ValueError("timestamps must have one value per sample")
        with self._lock:
            self._blocks.append(np.asarray(samples, dtype=np.float64) * self._scale)
            self._timestamps.append(np.asarray(timestamps, dtype=np.float64))

    def add_markers(self, markers: list[Marker]) -> None:
        """Append markers; their timestamps must be on the same clock as the samples."""
        with self._lock:
            self._markers.extend(markers)

    def to_raw(self) -> mne.io.RawArray:
        """Build an MNE :class:`~mne.io.RawArray` of everything recorded so far.

        Raises:
            ValueError: If no sample was recorded.
        """
        with self._lock:
            if not self._blocks:
                raise ValueError("nothing was recorded")
            data_uv = np.vstack(self._blocks)
            times = np.concatenate(self._timestamps)
            markers = list(self._markers)

        info = self.info
        mne_info = mne.create_info(
            ch_names=list(info.labels),
            sfreq=info.srate,
            ch_types=[_MNE_TYPES[c.type] for c in info.channels],
        )
        mne_info["subject_info"] = {"his_id": self.subject}
        mne_info["description"] = json.dumps({METADATA_KEY: self._metadata()})
        raw = mne.io.RawArray(data_uv.T * 1e-6, mne_info, verbose=False)
        raw.set_meas_date(self.started_at)

        positions = {
            label: np.array(POSITIONS_10_10[label])
            for label in info.labels
            if label in POSITIONS_10_10
        }
        if positions:
            raw.set_montage(
                mne.channels.make_dig_montage(ch_pos=positions, coord_frame="head"),
                on_missing="ignore",
                verbose=False,
            )
        raw.set_annotations(self._annotations(times, markers))
        return raw

    def save(self, path: str | Path | None = None, overwrite: bool = False) -> Path:
        """Write the recording to ``path`` (default: :func:`default_path`).

        The file name must end with ``_raw.fif`` (MNE convention); parent folders are
        created.

        Returns:
            The path written.

        Raises:
            ValueError: If nothing was recorded or the file name is not ``*_raw.fif``.
            FileExistsError: If the file exists and ``overwrite`` is false.
        """
        target = Path(path) if path is not None else default_path(self.subject)
        if not target.name.endswith("_raw.fif"):
            raise ValueError(f"recording file names must end with '_raw.fif', got {target.name}")
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} already exists")
        raw = self.to_raw()
        target.parent.mkdir(parents=True, exist_ok=True)
        raw.save(target, fmt="double", overwrite=overwrite, verbose=False)
        return target

    # --- Internals ------------------------------------------------------------

    def _metadata(self) -> dict[str, Any]:
        return {
            **self.info.model_dump(exclude={"channels"}),
            "spec_version": SPEC_VERSION,
            "software": "openmind-plug",
            "software_version": __version__,
        }

    def _annotations(
        self, times: npt.NDArray[np.float64], markers: list[Marker]
    ) -> mne.Annotations:
        period = 1.0 / self.info.srate
        onsets: list[float] = []
        descriptions: list[str] = []
        # Positions in the file are sample indices, so map each timestamp to the
        # index of the first sample at or after it.
        for marker in sorted(markers, key=lambda m: m.timestamp):
            index = int(np.searchsorted(times, marker.timestamp - period / 2))
            if index >= len(times):
                continue  # after the last sample
            kind = str(marker.payload.get("type", "event"))
            label = str(marker.payload.get("label", ""))
            onsets.append(index * period)
            descriptions.append(f"{kind}/{label}" if label else kind)
        gaps = np.flatnonzero(np.diff(times) > 1.5 * period)
        for gap in gaps:
            onsets.append((gap + 1) * period)
            descriptions.append(GAP_ANNOTATION)
        return mne.Annotations(onset=onsets, duration=[0.0] * len(onsets), description=descriptions)


def record_source(source: EEGSource, duration: float, subject: str) -> Recorder:
    """Record ``duration`` seconds of a source, read in real time.

    The source is started and stopped by this function.
    """
    import time

    recorder = Recorder(source.info, subject)
    n_target = round(duration * source.info.srate)
    with source:
        while recorder.n_samples < n_target:
            samples, timestamps = source.read(n_target - recorder.n_samples)
            recorder.add(samples, timestamps)
            recorder.add_markers(source.read_markers())
            if samples.shape[0] == 0:
                if getattr(source, "finished", False):
                    break
                time.sleep(0.01)
    recorder.add_markers(source.read_markers())
    return recorder


def record_stream(
    subject: str,
    duration: float | None = None,
    timeout: float = 5.0,
    source_id: str | None = None,
    stop_event: threading.Event | None = None,
) -> Recorder:
    """Record the standard LSL stream (EEG and markers) until stopped.

    Stops after ``duration`` seconds, when ``stop_event`` is set, or on Ctrl+C.

    Raises:
        StreamNotFoundError: If no stream is found within ``timeout``.
    """
    import pylsl

    from openmind_plug.inlet import EEGInlet

    validate_subject(subject)
    with EEGInlet(timeout=timeout, source_id=source_id) as inlet:
        recorder = Recorder(inlet.info, subject)
        start = pylsl.local_clock()
        try:
            while True:
                samples, timestamps = inlet.read(timeout=0.1)
                recorder.add(samples, timestamps)
                recorder.add_markers(inlet.read_markers())
                if stop_event is not None and stop_event.is_set():
                    break
                if duration is not None and pylsl.local_clock() - start >= duration:
                    break
        except KeyboardInterrupt:
            pass
        samples, timestamps = inlet.read()
        recorder.add(samples, timestamps)
        recorder.add_markers(inlet.read_markers())
    return recorder
