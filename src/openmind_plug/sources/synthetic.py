"""Synthetic EEG source: a realistic, controllable and reproducible test signal.

The signal of each channel, in microvolts, is the sum of:

- a **1/f background** (pink noise), independent per channel;
- **rhythm generators**, each a sinusoid with a slowly fluctuating amplitude, spread
  over nearby electrodes with a Gaussian spatial weight:

  - posterior **alpha** (8-12 Hz), much stronger with the eyes closed;
  - **mu** (8-12 Hz) and **beta** (13-30 Hz) over the left (C3), central (Cz) and right
    (C4) sensorimotor cortex;

- optional **artifacts**: eye blinks (slow, large, frontal), muscle bursts (broadband
  high frequency, lateral) and power-line noise (50 or 60 Hz);
- **jaw clenches** on demand (:meth:`SyntheticSource.clench`): short, strong bursts of
  muscle activity on every electrode, single or double, the gesture behind the
  ``action`` command.

**Event-related desynchronization (ERD).** Imagining a movement lowers the mu and
beta power over the opposite hemisphere. The current state drives this:
``left`` lowers the generators over C4, ``right`` over C3, ``feet`` over Cz, ``rest``
none. Gains move smoothly towards their target (time constant ``erd_tau_s``).

**Reproducibility.** For a given seed and the same sequence of state changes, the
signal only depends on the sample index, not on how it is split into reads: reading
100 then 150 samples gives exactly the same data as reading 250 at once.

**Synthetic subjects.** A :class:`SyntheticSubject` makes the simulated user individual:
rhythm frequencies, ERD depth per class (including "BCI-illiterate" users), electrode
contact gains, noise level and a small spatial mixing that changes the covariances.
Subjects are drawn deterministically from ``(seed, subject_id)``; an optional session
index adds a small session-to-session drift. Without a subject, the signal is exactly
the one of :meth:`SyntheticSubject.default`.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt
import pylsl
from pydantic import BaseModel, ConfigDict, Field
from scipy import signal

from openmind_plug.montage import DEFAULT_MONTAGE, Montage, get_montage, position
from openmind_plug.sources.base import (
    EEGSource,
    Marker,
    SampleArray,
    StreamInfo,
    TimestampArray,
)

logger = logging.getLogger(__name__)

State = Literal["rest", "left", "right", "feet"]
Eyes = Literal["open", "closed"]

#: Mental states the source can simulate.
STATES: Final[tuple[State, ...]] = ("rest", "left", "right", "feet")

#: Keyboard bindings used by interactive front-ends (see :meth:`SyntheticSource.handle_key`).
KEY_BINDINGS: Final[dict[str, str]] = {
    "r": "state: rest",
    "a": "state: left",
    "d": "state: right",
    "s": "state: feet",
    "e": "toggle eyes open / closed",
    "b": "trigger a blink",
    "j": "trigger a double jaw clench",
}

#: Lowest sampling rate supported: beta (up to 30 Hz) and muscle artifacts need headroom.
MIN_SRATE: Final[float] = 100.0

# IIR approximation of a 1/f power spectrum (J. O. Smith, "Spectral Audio Signal
# Processing", pink noise filter); accurate within ~0.3 dB over three decades.
_PINK_B: Final = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
_PINK_A: Final = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])

_SPATIAL_SIGMA_M: Final = 0.025  # spread of a rhythm generator around its centre
_BLINK_SIGMA_M: Final = 0.03  # spread of blinks around Fp1/Fp2
_ENVELOPE_TAU_S: Final = 0.5  # time constant of the amplitude fluctuations
_ENVELOPE_DEPTH: Final = 0.3  # relative standard deviation of the amplitude
_CLENCH_S: Final = (0.2, 0.3)  # duration of one clench burst
# Onset-to-onset interval of a double clench: always leaves a release of at least
# 0.3 s between the two bursts, as the gesture requires.
_CLENCH_GAP_S: Final = (0.6, 0.75)

#: Largest number of channels a :class:`SyntheticSubject` describes (gains and mixing).
MAX_SUBJECT_CHANNELS: Final = 32
#: Ranges drawn by :meth:`SyntheticSubject.sample`, as ``(low, high)``.
ALPHA_HZ_RANGE: Final = (8.5, 12.0)  # individual alpha frequency
MU_OFFSET_HZ: Final = 0.5  # mu is within this distance of alpha (and inside ALPHA_HZ_RANGE)
BETA_HZ_RANGE: Final = (18.0, 24.0)
ERD_DEPTH_RANGE: Final = (0.2, 0.7)  # relative power drop of a typical subject
ERD_DEPTH_WEAK_RANGE: Final = (0.05, 0.15)  # "BCI-illiterate" subjects
WEAK_ERD_PROBABILITY: Final = 0.15
CHANNEL_GAIN_RANGE: Final = (0.7, 1.3)  # electrode contact
NOISE_UV_RANGE: Final = (10.0, 20.0)  # RMS of the 1/f background
RHYTHM_AMPLITUDE_UV_RANGE: Final = (6.0, 14.0)  # mu amplitude (peak, at the generator)
MIXING_SPREAD: Final = 0.06  # std of the off-diagonal mixing terms
SESSION_GAIN_SPREAD: Final = 0.05  # per-channel gain drift between sessions (+/- 5 %)
SESSION_MIXING_SPREAD: Final = 0.015  # std of the mixing perturbation between sessions

_ERD_CLASSES: Final = ("left", "right", "feet")


class SyntheticConfig(BaseModel):
    """Amplitudes and artifact settings of :class:`SyntheticSource`.

    Amplitudes are in microvolts. Rhythm amplitudes are peak values on the electrode
    at the centre of the generator. All artifacts are disabled by default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    background_uv: float = Field(default=15.0, ge=0, description="RMS of the 1/f background.")
    alpha_open_uv: float = Field(default=4.0, ge=0, description="Alpha, eyes open.")
    alpha_closed_uv: float = Field(default=20.0, ge=0, description="Alpha, eyes closed.")
    mu_uv: float = Field(default=10.0, ge=0)
    beta_uv: float = Field(default=4.0, ge=0)
    erd_depth: float = Field(default=0.6, ge=0, le=1, description="Relative power drop.")
    erd_tau_s: float = Field(default=0.3, gt=0, description="ERD onset/offset time constant.")
    blink_rate_per_min: float = Field(default=0.0, ge=0)
    blink_uv: float = Field(default=120.0, ge=0)
    muscle_rate_per_min: float = Field(default=0.0, ge=0)
    muscle_uv: float = Field(default=20.0, ge=0, description="RMS during a burst.")
    # 120 µV puts a clench about 26 dB above the default background above 30 Hz,
    # like the deliberate clenches of public recordings.
    clench_uv: float = Field(default=120.0, ge=0, description="RMS during a jaw clench.")
    line_noise_uv: float = Field(default=0.0, ge=0)
    line_freq_hz: Literal[50, 60] = 50


@dataclass(frozen=True)
class SyntheticSubject:
    """Individual characteristics of a simulated user.

    ``None`` means "use the :class:`SyntheticConfig` value" (or, for ``channel_gain``
    and ``spatial_mixing``, no change), so that :meth:`default` reproduces the
    original signal exactly.

    Attributes:
        subject_id: Identifier, also part of the random draw key.
        alpha_hz: Individual alpha frequency (centre; each generator adds +/-0.5 Hz).
        mu_hz: Centre frequency of the mu generators.
        beta_hz: Centre frequency of the beta generators.
        erd_depth: Relative power drop per class (``left``, ``right``, ``feet``).
        channel_gain: Per-channel gain (electrode contact); the first
            ``n_channels`` values are used.
        noise_uv: RMS of the 1/f background, in microvolts.
        spatial_mixing: Square matrix close to the identity applied to the channels
            (volume conduction, reference); the leading block is used.
        rhythm_amplitude_uv: Peak mu amplitude; beta keeps its config ratio to mu.
    """

    subject_id: int = 0
    alpha_hz: float = 10.0
    mu_hz: float = 10.5
    beta_hz: float = 21.0
    erd_depth: Mapping[str, float] | None = None
    channel_gain: tuple[float, ...] | None = None
    noise_uv: float | None = None
    spatial_mixing: tuple[tuple[float, ...], ...] | None = None
    rhythm_amplitude_uv: float | None = None

    def __post_init__(self) -> None:
        if self.subject_id < 0:
            raise ValueError(f"subject_id must be >= 0, got {self.subject_id}")
        for name in ("alpha_hz", "mu_hz", "beta_hz"):
            if not 1.0 <= getattr(self, name) <= 40.0:
                raise ValueError(f"{name} must be within 1-40 Hz, got {getattr(self, name)}")
        if self.erd_depth is not None:
            if set(self.erd_depth) != set(_ERD_CLASSES):
                raise ValueError(f"erd_depth needs exactly the classes {list(_ERD_CLASSES)}")
            if not all(0.0 <= v <= 1.0 for v in self.erd_depth.values()):
                raise ValueError("erd_depth values must be within [0, 1]")
        if self.channel_gain is not None and not all(g > 0 for g in self.channel_gain):
            raise ValueError("channel_gain values must be > 0")
        if self.noise_uv is not None and self.noise_uv < 0:
            raise ValueError("noise_uv must be >= 0")
        if self.rhythm_amplitude_uv is not None and self.rhythm_amplitude_uv < 0:
            raise ValueError("rhythm_amplitude_uv must be >= 0")
        if self.spatial_mixing is not None:
            size = len(self.spatial_mixing)
            if size == 0 or any(len(row) != size for row in self.spatial_mixing):
                raise ValueError("spatial_mixing must be a non-empty square matrix")

    @classmethod
    def default(cls) -> SyntheticSubject:
        """The neutral subject: exactly the behaviour of a source with no subject."""
        return cls()

    @classmethod
    def sample(cls, subject_id: int, seed: int = 0) -> SyntheticSubject:
        """Draw a subject deterministically from ``(seed, subject_id)``.

        About 15 % of subjects are weak desynchronizers (``ERD_DEPTH_WEAK_RANGE``).
        Ranges are the module constants ``*_RANGE``.
        """
        if subject_id < 0 or seed < 0:
            raise ValueError("subject_id and seed must be >= 0")
        rng = np.random.default_rng((seed, subject_id))
        alpha = float(rng.uniform(*ALPHA_HZ_RANGE))
        mu = float(np.clip(alpha + rng.uniform(-MU_OFFSET_HZ, MU_OFFSET_HZ), *ALPHA_HZ_RANGE))
        beta = float(rng.uniform(*BETA_HZ_RANGE))
        weak = rng.random() < WEAK_ERD_PROBABILITY
        depth_range = ERD_DEPTH_WEAK_RANGE if weak else ERD_DEPTH_RANGE
        erd = {name: float(rng.uniform(*depth_range)) for name in _ERD_CLASSES}
        gain = rng.uniform(*CHANNEL_GAIN_RANGE, size=MAX_SUBJECT_CHANNELS)
        mixing = np.eye(MAX_SUBJECT_CHANNELS) + MIXING_SPREAD * rng.standard_normal(
            (MAX_SUBJECT_CHANNELS, MAX_SUBJECT_CHANNELS)
        )
        return cls(
            subject_id=subject_id,
            alpha_hz=alpha,
            mu_hz=mu,
            beta_hz=beta,
            erd_depth=erd,
            channel_gain=tuple(float(g) for g in gain),
            noise_uv=float(rng.uniform(*NOISE_UV_RANGE)),
            spatial_mixing=tuple(tuple(float(v) for v in row) for row in mixing),
            rhythm_amplitude_uv=float(rng.uniform(*RHYTHM_AMPLITUDE_UV_RANGE)),
        )

    def with_session(self, session: int, seed: int = 0) -> SyntheticSubject:
        """Return this subject as seen in a given session (small drift).

        Gains drift by up to +/-5 % per channel and the mixing is slightly perturbed,
        drawn from ``(seed, subject_id, session)``: what cross-session alignment
        corrects.
        """
        if session < 0 or seed < 0:
            raise ValueError("session and seed must be >= 0")
        rng = np.random.default_rng((seed, self.subject_id, session))
        size = MAX_SUBJECT_CHANNELS
        drift = rng.uniform(1 - SESSION_GAIN_SPREAD, 1 + SESSION_GAIN_SPREAD, size=size)
        perturbation = SESSION_MIXING_SPREAD * rng.standard_normal((size, size))
        base_gain = np.ones(size)
        if self.channel_gain is not None:
            base_gain[: len(self.channel_gain)] = self.channel_gain
        base_mixing = np.eye(size)
        if self.spatial_mixing is not None:
            n = len(self.spatial_mixing)
            base_mixing[:n, :n] = np.asarray(self.spatial_mixing)
        return replace(
            self,
            channel_gain=tuple(float(g) for g in base_gain * drift),
            spatial_mixing=tuple(
                tuple(float(v) for v in row) for row in base_mixing + perturbation
            ),
        )

    def _gain(self, n_channels: int) -> npt.NDArray[np.float64] | None:
        if self.channel_gain is None:
            return None
        if len(self.channel_gain) < n_channels:
            raise ValueError(
                f"subject describes {len(self.channel_gain)} channels, montage has {n_channels}"
            )
        return np.asarray(self.channel_gain[:n_channels], dtype=np.float64)

    def _mixing(self, n_channels: int) -> npt.NDArray[np.float64] | None:
        if self.spatial_mixing is None:
            return None
        if len(self.spatial_mixing) < n_channels:
            raise ValueError(
                f"subject mixes {len(self.spatial_mixing)} channels, montage has {n_channels}"
            )
        matrix = np.asarray(self.spatial_mixing, dtype=np.float64)
        return matrix[:n_channels, :n_channels]


@dataclass
class _Rhythm:
    """One oscillating generator spread over several channels."""

    kind: Literal["alpha", "mu", "beta"]
    site: Literal["posterior", "left", "central", "right"]
    freq_hz: float
    amplitude_uv: float
    phase: float
    weights: npt.NDArray[np.float64]
    gain: float = 1.0


@dataclass
class _Event:
    """A transient artifact, pre-rendered for all channels."""

    start: int
    waveform: npt.NDArray[np.float64]  # (n_samples, n_channels)

    @property
    def stop(self) -> int:
        return self.start + int(self.waveform.shape[0])


class SyntheticSource(EEGSource):
    """Realistic synthetic EEG with controllable motor-imagery state.

    Args:
        montage: Montage name or object; defines the channels (default ``"C"``).
        srate: Sampling rate in Hz, at least :data:`MIN_SRATE`.
        seed: Random seed; the same seed gives the same signal.
        config: Amplitudes and artifacts; defaults to :class:`SyntheticConfig`.
        emit_markers: Produce a marker at each state or eyes change and at each
            artifact, as ground truth (see :meth:`read_markers`).
        clock: Time source used by :meth:`read`; defaults to ``pylsl.local_clock``.
        subject: Simulated user: a :class:`SyntheticSubject`, or an ``int`` id drawn with
            :meth:`SyntheticSubject.sample` (``seed`` ``None`` counts as 0, so the same
            id is always the same user). Default: :meth:`SyntheticSubject.default`.
        session: Optional session index; adds the small session-to-session drift of
            :meth:`SyntheticSubject.with_session`.

    Example::

        source = SyntheticSource(seed=42)
        with source:
            source.set_state("left")
            samples, timestamps = source.read(max_samples=64)
    """

    def __init__(
        self,
        montage: str | Montage = DEFAULT_MONTAGE,
        srate: float = 250.0,
        seed: int | None = None,
        config: SyntheticConfig | None = None,
        emit_markers: bool = True,
        clock: Callable[[], float] = pylsl.local_clock,
        subject: SyntheticSubject | int | None = None,
        session: int | None = None,
    ) -> None:
        super().__init__()
        self._montage = get_montage(montage) if isinstance(montage, str) else montage
        self.config = config or SyntheticConfig()
        subject_seed = seed if seed is not None else 0
        if isinstance(subject, int):
            subject = SyntheticSubject.sample(subject, subject_seed)
        resolved = subject or SyntheticSubject.default()
        if session is not None:
            resolved = resolved.with_session(session, subject_seed)
        self.subject = resolved
        if subject is not None or session is not None:
            logger.info("synthetic subject %d, session %s", resolved.subject_id, session)
        if not srate >= MIN_SRATE or not math.isfinite(srate):
            raise ValueError(f"srate must be finite and >= {MIN_SRATE} Hz, got {srate}")
        self._info = StreamInfo(
            srate=srate,
            channels=self._montage.channels(),
            manufacturer="OpenMind",
            model="synthetic",
            driver="synthetic",
            reference="none",
            montage=self._montage.name,
        )
        self._emit_markers = emit_markers
        self._clock = clock
        self._lock = threading.Lock()

        positions = np.array(list(self._montage.positions().values()), dtype=np.float64)
        n_channels = positions.shape[0]
        seeds = np.random.SeedSequence(seed).spawn(6)
        rngs = [np.random.default_rng(s) for s in seeds]
        setup_rng, self._rng_background, self._rng_envelope = rngs[:3]
        self._rng_events, self._rng_shapes, line_rng = rngs[3:]

        # Background: unit-variance white noise through the pink filter, rescaled.
        impulse = signal.lfilter(_PINK_B, _PINK_A, np.r_[1.0, np.zeros(20_000)])
        noise_uv = self.subject.noise_uv
        if noise_uv is None:
            noise_uv = self.config.background_uv
        self._pink_scale = noise_uv / float(np.sqrt(np.sum(impulse**2)))
        self._pink_zi = np.zeros((len(_PINK_A) - 1, n_channels))

        self._channel_gain = self.subject._gain(n_channels)
        self._mixing = self.subject._mixing(n_channels)
        self._rhythms = self._make_rhythms(positions, setup_rng)
        self._envelope_rho = math.exp(-1.0 / (srate * _ENVELOPE_TAU_S))
        self._envelope_zi = np.zeros((1, len(self._rhythms)))
        self._gain_decay = math.exp(-1.0 / (srate * self.config.erd_tau_s))

        self._line_weights: npt.NDArray[np.float64] = np.asarray(
            line_rng.uniform(0.5, 1.5, size=n_channels), dtype=np.float64
        )
        self._line_phase = float(line_rng.uniform(0, 2 * math.pi))

        # Blinks originate at the eyes, just below the frontal poles Fp1/Fp2.
        poles = np.array([position("Fp1"), position("Fp2")])
        dist = np.min(np.linalg.norm(positions[:, None, :] - poles[None], axis=2), axis=1)
        self._blink_weights = np.exp(-(dist**2) / (2 * _BLINK_SIGMA_M**2))
        self._muscle_weights = 0.3 + 0.7 * np.clip(np.abs(positions[:, 0]) / 0.085, 0, 1)
        self._muscle_sos = signal.butter(4, 30.0, btype="highpass", fs=srate, output="sos")
        # Jaw muscles (masseter, temporalis) reach every scalp electrode, more so the
        # lateral ones.
        self._clench_weights = 0.6 + 0.4 * np.clip(np.abs(positions[:, 0]) / 0.085, 0, 1)

        self._state: State = "rest"
        self._eyes: Eyes = "open"
        self._applied: tuple[State, Eyes] | None = None
        self._events: list[_Event] = []
        self._unplaced_markers: list[tuple[int, dict[str, Any]]] = []
        self._pending_markers: list[Marker] = []
        self._produced = 0
        self._t0 = 0.0
        self._set_gain_targets(initial=True)

    # --- Public API -----------------------------------------------------------

    @property
    def info(self) -> StreamInfo:
        return self._info

    @property
    def state(self) -> State:
        """Current mental state."""
        return self._state

    @property
    def eyes(self) -> Eyes:
        """Current eyes state."""
        return self._eyes

    @property
    def n_produced(self) -> int:
        """Number of samples generated so far."""
        return self._produced

    def set_state(self, state: str) -> None:
        """Set the simulated mental state, applied from the next generated sample.

        Raises:
            ValueError: If ``state`` is not one of :data:`STATES`.
        """
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}; expected one of {list(STATES)}")
        with self._lock:
            self._state = state

    def set_eyes(self, eyes: str) -> None:
        """Open or close the simulated eyes (``"open"`` or ``"closed"``)."""
        if eyes not in ("open", "closed"):
            raise ValueError(f"eyes must be 'open' or 'closed', got {eyes!r}")
        with self._lock:
            self._eyes = cast(Eyes, eyes)

    def blink(self) -> None:
        """Trigger a blink at the next generated sample."""
        with self._lock:
            self._add_blink(self._produced)

    def clench(self, count: int = 2) -> None:
        """Trigger a jaw clench (``count=1``) or a double clench at the next sample.

        Each clench is a 0.2-0.3 s burst of muscle activity above 30 Hz on every
        electrode; the two clenches of a double clench start 0.6-0.75 s apart, with
        a release of at least 0.3 s between them. A
        ``jaw_clench`` artifact marker with the ``count`` marks the first one.

        Raises:
            ValueError: If ``count`` is not 1 or 2.
        """
        if count not in (1, 2):
            raise ValueError(f"count must be 1 or 2, got {count}")
        with self._lock:
            self._add_clench(self._produced, count)

    def handle_key(self, key: str) -> bool:
        """Apply a keyboard command from :data:`KEY_BINDINGS`.

        Returns:
            Whether the key was recognised.
        """
        key = key.lower()
        states: dict[str, State] = {"r": "rest", "a": "left", "d": "right", "s": "feet"}
        if key in states:
            self.set_state(states[key])
        elif key == "e":
            self.set_eyes("closed" if self._eyes == "open" else "open")
        elif key == "b":
            self.blink()
        elif key == "j":
            self.clench(2)
        else:
            return False
        return True

    def generate(self, n_samples: int) -> SampleArray:
        """Generate the next ``n_samples`` samples, regardless of the clock.

        Useful offline (tests, datasets). In real time, use :meth:`read`, which paces
        the output on the clock; mixing both shifts the real-time output.

        Returns:
            ``float32`` array of shape ``(n_samples, n_channels)``, in microvolts.
        """
        if n_samples < 0:
            raise ValueError(f"n_samples must be >= 0, got {n_samples}")
        with self._lock:
            return self._generate(n_samples).astype(np.float32)

    def read_markers(self) -> list[Marker]:
        with self._lock:
            markers, self._pending_markers = self._pending_markers, []
        return markers

    # --- EEGSource hooks ------------------------------------------------------

    def _start(self) -> None:
        # Anchor the clock so that the stream continues where it stopped.
        self._t0 = self._clock() - self._produced / self._info.srate

    def _stop(self) -> None:
        pass

    def _read(self, max_samples: int) -> tuple[SampleArray, TimestampArray]:
        srate = self._info.srate
        with self._lock:
            # The small epsilon absorbs float rounding (0.4 s * 250 Hz = 99.99999...).
            due = math.floor((self._clock() - self._t0) * srate + 1e-6) - self._produced
            n = min(max(due, 0), max_samples)
            first = self._produced
            samples = self._generate(n).astype(np.float32)
        timestamps = self._t0 + (first + np.arange(n, dtype=np.float64)) / srate
        return samples, timestamps

    # --- Generation -----------------------------------------------------------

    def _generate(self, n: int) -> npt.NDArray[np.float64]:
        n_channels = self._info.n_channels
        if n == 0:
            return np.zeros((0, n_channels))
        self._apply_changes()
        srate = self._info.srate
        cfg = self.config
        index = self._produced + np.arange(n, dtype=np.float64)
        t: npt.NDArray[np.float64] = index / srate

        white = self._rng_background.standard_normal((n, n_channels))
        out: npt.NDArray[np.float64]
        out, self._pink_zi = signal.lfilter(_PINK_B, _PINK_A, white, axis=0, zi=self._pink_zi)
        out *= self._pink_scale

        rho = self._envelope_rho
        noise = self._rng_envelope.standard_normal((n, len(self._rhythms)))
        envelope, self._envelope_zi = signal.lfilter(
            [math.sqrt(1 - rho**2)], [1.0, -rho], noise, axis=0, zi=self._envelope_zi
        )
        decay = self._gain_decay ** np.arange(1, n + 1)
        for i, rhythm in enumerate(self._rhythms):
            target = self._gain_targets[i]
            gain = target + (rhythm.gain - target) * decay
            rhythm.gain = float(gain[-1])
            amplitude = (
                rhythm.amplitude_uv * gain * np.maximum(0.0, 1.0 + _ENVELOPE_DEPTH * envelope[:, i])
            )
            wave = amplitude * np.sin(2 * math.pi * rhythm.freq_hz * t + rhythm.phase)
            out += wave[:, None] * rhythm.weights[None, :]

        if cfg.line_noise_uv > 0:
            line = np.sin(2 * math.pi * cfg.line_freq_hz * t + self._line_phase)
            out += cfg.line_noise_uv * line[:, None] * self._line_weights[None, :]

        self._spawn_events(n)
        stop = self._produced + n
        for event in self._events:
            lo, hi = max(event.start, self._produced), min(event.stop, stop)
            if lo < hi:
                out[lo - self._produced : hi - self._produced] += event.waveform[
                    lo - event.start : hi - event.start
                ]
        self._events = [e for e in self._events if e.stop > stop]
        self._produced = stop
        self._place_markers()
        if self._mixing is not None:
            out = out @ self._mixing.T
        if self._channel_gain is not None:
            out = out * self._channel_gain
        return out

    def _apply_changes(self) -> None:
        current = (self._state, self._eyes)
        if current == self._applied:
            return
        previous = self._applied
        self._applied = current
        self._set_gain_targets()
        if previous is None:
            return
        if previous[0] != current[0]:
            self._marker({"type": "state", "label": current[0]})
        if previous[1] != current[1]:
            self._marker({"type": "eyes", "label": current[1]})

    def _set_gain_targets(self, initial: bool = False) -> None:
        erd_site = {"rest": None, "left": "right", "right": "left", "feet": "central"}
        site = erd_site[self._state]
        cfg = self.config
        alpha_max = max(cfg.alpha_open_uv, cfg.alpha_closed_uv)
        alpha_now = cfg.alpha_closed_uv if self._eyes == "closed" else cfg.alpha_open_uv
        targets = []
        for rhythm in self._rhythms:
            if rhythm.kind == "alpha":
                targets.append(alpha_now / alpha_max if alpha_max > 0 else 1.0)
            elif rhythm.site == site:
                # erd_depth is a power drop: amplitude scales with its square root.
                depth = self.subject.erd_depth
                depth_now = cfg.erd_depth if depth is None else depth[self._state]
                targets.append(math.sqrt(1.0 - depth_now))
            else:
                targets.append(1.0)
        self._gain_targets = targets
        if initial:
            for rhythm, target in zip(self._rhythms, targets, strict=True):
                rhythm.gain = target

    def _make_rhythms(
        self, positions: npt.NDArray[np.float64], rng: np.random.Generator
    ) -> list[_Rhythm]:
        cfg = self.config
        subject = self.subject
        mu_amplitude = (
            cfg.mu_uv if subject.rhythm_amplitude_uv is None else subject.rhythm_amplitude_uv
        )
        beta_amplitude = cfg.beta_uv
        if subject.rhythm_amplitude_uv is not None and cfg.mu_uv > 0:
            beta_amplitude = cfg.beta_uv * subject.rhythm_amplitude_uv / cfg.mu_uv
        rhythms: list[_Rhythm] = []
        # Alpha: weight grows from 0 at the central line to 1 over the occipital pole.
        posterior = np.clip((-positions[:, 1] - 0.04) / 0.07, 0.0, 1.0)
        rhythms.append(
            _Rhythm(
                kind="alpha",
                site="posterior",
                freq_hz=subject.alpha_hz + rng.uniform(-0.5, 0.5),
                amplitude_uv=max(cfg.alpha_open_uv, cfg.alpha_closed_uv),
                phase=rng.uniform(0, 2 * math.pi),
                weights=posterior,
            )
        )
        centres = {"left": "C3", "central": "Cz", "right": "C4"}
        for kind, base_hz, amplitude in (
            ("mu", subject.mu_hz, mu_amplitude),
            ("beta", subject.beta_hz, beta_amplitude),
        ):
            for site, label in centres.items():
                centre = np.array(position(label))
                dist = np.linalg.norm(positions - centre, axis=1)
                weights = np.exp(-(dist**2) / (2 * _SPATIAL_SIGMA_M**2))
                rhythms.append(
                    _Rhythm(
                        kind=cast(Literal["mu", "beta"], kind),
                        site=cast(Literal["left", "central", "right"], site),
                        freq_hz=base_hz + rng.uniform(-0.5, 0.5),
                        amplitude_uv=amplitude,
                        phase=rng.uniform(0, 2 * math.pi),
                        weights=weights,
                    )
                )
        return rhythms

    # --- Artifacts ------------------------------------------------------------

    def _spawn_events(self, n: int) -> None:
        cfg = self.config
        srate = self._info.srate
        draws = self._rng_events.random((n, 2))
        p_blink = cfg.blink_rate_per_min / 60.0 / srate
        p_muscle = cfg.muscle_rate_per_min / 60.0 / srate
        # Visit events in time order so that random draws do not depend on block sizes.
        hits = np.argwhere(draws < np.array([p_blink, p_muscle]))
        for offset, kind in hits:
            start = self._produced + int(offset)
            if kind == 0:
                self._add_blink(start)
            else:
                self._add_muscle(start)

    def _add_blink(self, start: int) -> None:
        rng, srate = self._rng_shapes, self._info.srate
        length = max(int(rng.uniform(0.25, 0.4) * srate), 3)
        amplitude = self.config.blink_uv * rng.uniform(0.8, 1.2)
        shape = amplitude * signal.windows.hann(length)
        self._events.append(_Event(start, shape[:, None] * self._blink_weights[None, :]))
        self._marker({"type": "artifact", "label": "blink"}, index=start)

    def _add_muscle(self, start: int) -> None:
        rng, srate = self._rng_shapes, self._info.srate
        length = max(int(rng.uniform(0.5, 1.5) * srate), 8)
        white = rng.standard_normal((length, len(self._muscle_weights)))
        noise = signal.sosfilt(self._muscle_sos, white, axis=0)
        noise /= np.std(noise, axis=0, keepdims=True) + 1e-12
        taper = signal.windows.tukey(length, alpha=0.3)
        waveform = self.config.muscle_uv * noise * taper[:, None] * self._muscle_weights[None, :]
        self._events.append(_Event(start, waveform))
        self._marker({"type": "artifact", "label": "muscle"}, index=start)

    def _add_clench(self, start: int, count: int) -> None:
        rng, srate = self._rng_shapes, self._info.srate
        onset = start
        for _ in range(count):
            length = max(int(rng.uniform(*_CLENCH_S) * srate), 8)
            white = rng.standard_normal((length, len(self._clench_weights)))
            noise = signal.sosfilt(self._muscle_sos, white, axis=0)
            noise /= np.std(noise, axis=0, keepdims=True) + 1e-12
            taper = signal.windows.tukey(length, alpha=0.3)
            amplitude = self.config.clench_uv * rng.uniform(0.8, 1.2)
            waveform = amplitude * noise * taper[:, None] * self._clench_weights[None, :]
            self._events.append(_Event(onset, waveform))
            onset += int(rng.uniform(*_CLENCH_GAP_S) * srate)
        self._marker({"type": "artifact", "label": "jaw_clench", "count": count}, index=start)

    def _marker(self, payload: dict[str, Any], index: int | None = None) -> None:
        """Queue a marker for the sample at ``index`` (default: the next sample)."""
        if self._emit_markers:
            index = self._produced if index is None else index
            self._unplaced_markers.append((index, {**payload, "source": "synthetic"}))

    def _place_markers(self) -> None:
        """Timestamp the queued markers whose sample has now been generated."""
        srate = self._info.srate
        ready: list[tuple[int, dict[str, Any]]] = []
        waiting: list[tuple[int, dict[str, Any]]] = []
        for index, payload in self._unplaced_markers:
            (ready if index < self._produced else waiting).append((index, payload))
        self._unplaced_markers = waiting
        for index, payload in sorted(ready, key=lambda item: item[0]):
            self._pending_markers.append(Marker(self._t0 + index / srate, payload))
