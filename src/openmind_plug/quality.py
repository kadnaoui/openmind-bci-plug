"""Signal quality: per-channel checks and the eyes open / closed alpha test.

:func:`assess` looks at a short window of samples (a few seconds) and flags, for each
channel:

- **flat**: almost no signal (disconnected electrode, channel set to zero);
- **saturated**: the amplifier is clipping (samples stuck at the window extremes,
  or beyond ``saturation_uv`` when the amplifier range is known);
- **line noise**: a strong peak at the power-line frequency (bad contact);
- **abnormal variance**: much noisier or much quieter than the other channels.

:func:`alpha_test` compares alpha power between eyes open and eyes closed: a working
posterior electrode shows a clear increase when the eyes close (Berger effect).

All checks run on high-passed data (1 Hz), so the large DC offsets of many
amplifiers do not hide real problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from scipy import signal

Status = Literal["good", "flat", "saturated", "line noise", "noisy", "weak"]

#: Below this standard deviation (µV, after high-pass) a channel is flat.
FLAT_STD_UV: Final = 0.5
#: Fraction of samples stuck at an extreme above which a channel is saturated.
SATURATION_FRACTION: Final = 0.05
#: Line peak over its spectral neighbourhood above which a channel is noisy (10 dB).
LINE_NOISE_RATIO: Final = 10.0
#: Standard deviation over the median of all channels above which a channel is noisy.
NOISY_FACTOR: Final = 5.0
#: Alpha power increase (closed / open) above which the alpha test passes. Chance
#: alone reaches about x1.5 on the best of 8 channels with 30 s phases, while a real
#: eyes-closed effect is typically x2 to x10 on posterior electrodes.
ALPHA_GAIN_THRESHOLD: Final = 2.0
#: Shortest phase giving a reliable alpha test, in seconds.
ALPHA_MIN_RELIABLE_PHASE_S: Final = 20.0
#: Alpha band in Hz.
ALPHA_BAND: Final = (8.0, 12.0)


@dataclass(frozen=True)
class ChannelQuality:
    """Quality of one channel over a window."""

    label: str
    status: Status
    std_uv: float
    line_noise_ratio: float
    saturated_fraction: float
    impedance_kohm: float | None = None

    @property
    def ok(self) -> bool:
        """Whether no problem was found."""
        return self.status == "good"


def assess(
    samples: npt.ArrayLike,
    srate: float,
    labels: tuple[str, ...] | list[str],
    line_freq: float = 50.0,
    saturation_uv: float | None = None,
    impedances_kohm: dict[str, float] | None = None,
) -> list[ChannelQuality]:
    """Assess each channel over a window of samples.

    Args:
        samples: Microvolts, shape ``(n_samples, n_channels)``; at least one second.
        srate: Sampling rate in Hz.
        labels: Channel labels, in column order.
        line_freq: Power-line frequency (50 Hz in Europe, 60 Hz in the Americas).
        saturation_uv: Amplifier range in µV, if known; samples at or beyond it count
            as saturated. Without it, clipping is detected as samples stuck at the
            window extremes.
        impedances_kohm: Electrode impedances by label, when the driver measures them.

    Returns:
        One :class:`ChannelQuality` per channel, in column order. The worst problem
        wins: flat, then saturated, then line noise, then abnormal variance.

    Raises:
        ValueError: If the shapes do not match or the window is shorter than 1 s.
    """
    raw = np.asarray(samples, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(labels):
        raise ValueError(f"expected samples of shape (n, {len(labels)}), got {raw.shape}")
    if raw.shape[0] < srate:
        raise ValueError("the window must be at least 1 s long")

    filtered = _highpass(raw, srate)
    std = filtered.std(axis=0)
    saturated = _saturated_fraction(raw, saturation_uv)
    line = _line_noise_ratio(filtered, srate, line_freq)
    active = std[std >= FLAT_STD_UV]
    median = float(np.median(active)) if active.size else 0.0

    report: list[ChannelQuality] = []
    for i, label in enumerate(labels):
        status: Status = "good"
        if std[i] < FLAT_STD_UV:
            status = "flat"
        elif saturated[i] > SATURATION_FRACTION:
            status = "saturated"
        elif line[i] > LINE_NOISE_RATIO:
            status = "line noise"
        elif active.size >= 3 and std[i] > NOISY_FACTOR * median:
            status = "noisy"
        elif active.size >= 3 and std[i] < median / NOISY_FACTOR:
            status = "weak"
        report.append(
            ChannelQuality(
                label=label,
                status=status,
                std_uv=float(std[i]),
                line_noise_ratio=float(line[i]),
                saturated_fraction=float(saturated[i]),
                impedance_kohm=(impedances_kohm or {}).get(label),
            )
        )
    return report


@dataclass(frozen=True)
class AlphaReport:
    """Result of :func:`alpha_test`."""

    gains: dict[str, float]  # alpha power closed / open, per channel
    best_channel: str
    best_gain: float
    passed: bool


def alpha_test(
    eyes_open: npt.ArrayLike,
    eyes_closed: npt.ArrayLike,
    srate: float,
    labels: tuple[str, ...] | list[str],
    threshold: float = ALPHA_GAIN_THRESHOLD,
) -> AlphaReport:
    """Compare alpha power (8-12 Hz) between eyes open and eyes closed.

    Args:
        eyes_open: Microvolts, shape ``(n_samples, n_channels)``, eyes open.
        eyes_closed: Same, eyes closed. 30 s of each is recommended; below
            :data:`ALPHA_MIN_RELIABLE_PHASE_S` chance fluctuations can pass the test.
        srate: Sampling rate in Hz.
        labels: Channel labels, in column order.
        threshold: Minimum gain on the best channel for the test to pass.
    """
    open_power = band_power(eyes_open, srate, ALPHA_BAND)
    closed_power = band_power(eyes_closed, srate, ALPHA_BAND)
    gains = closed_power / np.maximum(open_power, 1e-12)
    best = int(np.argmax(gains))
    return AlphaReport(
        gains={label: float(gain) for label, gain in zip(labels, gains, strict=True)},
        best_channel=labels[best],
        best_gain=float(gains[best]),
        passed=bool(gains[best] >= threshold),
    )


def band_power(
    samples: npt.ArrayLike, srate: float, band: tuple[float, float]
) -> npt.NDArray[np.float64]:
    """Mean power spectral density (µV²/Hz) per channel in ``band``, by Welch's method."""
    data = np.asarray(samples, dtype=np.float64)
    freqs, psd = signal.welch(data, fs=srate, nperseg=min(data.shape[0], int(2 * srate)), axis=0)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.asarray(psd[mask].mean(axis=0), dtype=np.float64)


def _highpass(data: npt.NDArray[np.float64], srate: float) -> npt.NDArray[np.float64]:
    sos = signal.butter(2, 1.0, btype="highpass", fs=srate, output="sos")
    return np.asarray(signal.sosfiltfilt(sos, data - data.mean(axis=0), axis=0))


def _saturated_fraction(
    data: npt.NDArray[np.float64], saturation_uv: float | None
) -> npt.NDArray[np.float64]:
    if saturation_uv is not None:
        return np.asarray(np.mean(np.abs(data) >= saturation_uv, axis=0), dtype=np.float64)
    # Clipping shows as many samples equal to the window extremes.
    low, high = data.min(axis=0), data.max(axis=0)
    span = np.maximum(high - low, 1e-9)
    tolerance = 1e-6 * span
    at_low = np.abs(data - low) <= tolerance
    at_high = np.abs(data - high) <= tolerance
    fraction = np.maximum(at_low.mean(axis=0), at_high.mean(axis=0))
    # A window of identical values is flat, not saturated.
    return np.where(high - low > 0, fraction, 0.0)


def _line_noise_ratio(
    data: npt.NDArray[np.float64], srate: float, line_freq: float
) -> npt.NDArray[np.float64]:
    n_channels = data.shape[1]
    if line_freq + 10 >= srate / 2:
        return np.zeros(n_channels)  # line frequency not observable at this rate
    freqs, psd = signal.welch(data, fs=srate, nperseg=min(data.shape[0], int(srate)), axis=0)
    peak = psd[np.abs(freqs - line_freq) <= 1.0].mean(axis=0)
    near = (np.abs(freqs - line_freq) >= 3.0) & (np.abs(freqs - line_freq) <= 10.0)
    floor = np.median(psd[near], axis=0)
    return np.asarray(peak / np.maximum(floor, 1e-12), dtype=np.float64)
