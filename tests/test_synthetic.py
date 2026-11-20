"""Tests for the synthetic EEG source."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError
from scipy.signal import butter, sosfilt, welch

from openmind_plug.montage import get_montage
from openmind_plug.sources import EEGSource
from openmind_plug.sources.synthetic import (
    KEY_BINDINGS,
    MIN_SRATE,
    STATES,
    SyntheticConfig,
    SyntheticSource,
)

FS = 250


def band_power(x: npt.NDArray[np.float32], lo: float, hi: float) -> npt.NDArray[np.float64]:
    """Mean Welch power density per channel in [lo, hi] Hz."""
    freqs, psd = welch(x, fs=FS, nperseg=512, axis=0)
    mask = (freqs >= lo) & (freqs <= hi)
    return np.asarray(psd[mask].mean(axis=0), dtype=np.float64)


def channel(source: SyntheticSource, label: str) -> int:
    return source.info.labels.index(label)


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# --- Metadata and configuration -----------------------------------------------


def test_info_matches_montage() -> None:
    source = SyntheticSource(montage="B", srate=500)
    assert isinstance(source, EEGSource)
    assert source.info.labels == get_montage("B").labels
    assert source.info.srate == 500
    assert source.info.montage == "B"
    assert source.info.model == "synthetic"
    assert all(c.unit == "uV" for c in source.info.channels)


def test_default_montage_is_a() -> None:
    assert SyntheticSource().info.montage == "A"


@pytest.mark.parametrize("srate", [MIN_SRATE - 1, 0, float("nan"), float("inf")])
def test_rejects_low_or_invalid_srate(srate: float) -> None:
    with pytest.raises(ValueError, match="srate"):
        SyntheticSource(srate=srate)


def test_config_validation() -> None:
    with pytest.raises(ValidationError):
        SyntheticConfig(erd_depth=1.5)
    with pytest.raises(ValidationError):
        SyntheticConfig(line_freq_hz=55)  # type: ignore[arg-type]


def test_rejects_unknown_state_and_eyes() -> None:
    source = SyntheticSource()
    with pytest.raises(ValueError, match="unknown state"):
        source.set_state("jump")
    with pytest.raises(ValueError, match="eyes"):
        source.set_eyes("half")


# --- Reproducibility ------------------------------------------------------------


def test_same_seed_same_signal() -> None:
    a = SyntheticSource(seed=7).generate(500)
    b = SyntheticSource(seed=7).generate(500)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ() -> None:
    a = SyntheticSource(seed=1).generate(500)
    b = SyntheticSource(seed=2).generate(500)
    assert not np.allclose(a, b)


def test_signal_does_not_depend_on_read_sizes() -> None:
    config = SyntheticConfig(blink_rate_per_min=30, muscle_rate_per_min=10, line_noise_uv=5)
    whole = SyntheticSource(seed=3, config=config)
    split = SyntheticSource(seed=3, config=config)
    expected = whole.generate(2000)
    parts = [split.generate(n) for n in (1, 99, 400, 700, 800)]
    np.testing.assert_allclose(np.vstack(parts), expected, rtol=0, atol=1e-3)


def test_output_shape_and_dtype() -> None:
    samples = SyntheticSource().generate(10)
    assert samples.shape == (10, 8)
    assert samples.dtype == np.float32
    assert SyntheticSource().generate(0).shape == (0, 8)


# --- Physiology -----------------------------------------------------------------


def test_background_amplitude_is_realistic() -> None:
    config = SyntheticConfig(alpha_open_uv=0, alpha_closed_uv=0, mu_uv=0, beta_uv=0)
    samples = SyntheticSource(seed=0, config=config).generate(30 * FS)
    rms = samples.std(axis=0)
    assert np.all((rms > 10) & (rms < 30)), rms


def test_background_is_one_over_f() -> None:
    config = SyntheticConfig(alpha_open_uv=0, alpha_closed_uv=0, mu_uv=0, beta_uv=0)
    samples = SyntheticSource(seed=0, config=config).generate(60 * FS)
    low, high = band_power(samples, 2, 4), band_power(samples, 20, 40)
    # 1/f: a tenfold frequency increase divides power by about ten.
    ratio = low / high
    assert np.all((ratio > 4) & (ratio < 25)), ratio


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_left_imagery_desynchronizes_mu_on_c4(seed: int) -> None:
    source = SyntheticSource(seed=seed)
    rest = source.generate(60 * FS)
    source.set_state("left")
    source.generate(FS)  # let the ERD settle
    left = source.generate(60 * FS)

    ratio = band_power(left, 8, 13) / band_power(rest, 8, 13)
    c3, c4 = ratio[channel(source, "C3")], ratio[channel(source, "C4")]
    assert c4 < 0.7, ratio
    assert c4 < c3 - 0.15, ratio


@pytest.mark.parametrize(("state", "site"), [("right", "C3"), ("feet", "Cz")])
def test_other_states_desynchronize_their_site(state: str, site: str) -> None:
    source = SyntheticSource(seed=0)
    rest = source.generate(60 * FS)
    source.set_state(state)
    source.generate(FS)
    active = source.generate(60 * FS)
    ratio = band_power(active, 8, 13) / band_power(rest, 8, 13)
    assert ratio[channel(source, site)] == ratio.min(), ratio
    assert ratio[channel(source, site)] < 0.7, ratio


def test_eyes_closed_increase_posterior_alpha() -> None:
    source = SyntheticSource(montage="B", seed=0)
    opened = source.generate(20 * FS)
    source.set_eyes("closed")
    source.generate(FS)
    closed = source.generate(20 * FS)
    gain = band_power(closed, 8, 12) / band_power(opened, 8, 12)
    assert gain[channel(source, "O1")] > 4
    assert gain[channel(source, "O2")] > 4
    assert gain[channel(source, "Cz")] < 1.5


# --- Artifacts ------------------------------------------------------------------


def test_blink_is_detectable_on_fp1() -> None:
    source = SyntheticSource(seed=0)
    baseline = source.generate(2 * FS)
    source.blink()
    with_blink = source.generate(FS)
    fp1, c4 = channel(source, "Fp1"), channel(source, "C4")
    assert with_blink[:, fp1].max() > 80
    assert with_blink[:, fp1].max() > 5 * baseline[:, fp1].std()
    assert np.abs(with_blink[:, c4]).max() < 5 * baseline[:, c4].std()


def test_random_blinks_follow_rate() -> None:
    config = SyntheticConfig(blink_rate_per_min=20)
    source = SyntheticSource(seed=0, config=config)
    source.generate(60 * FS)
    blinks = [m for m in source.read_markers() if m.payload["label"] == "blink"]
    assert 10 <= len(blinks) <= 32


def test_muscle_burst_adds_high_frequency_power() -> None:
    quiet = SyntheticSource(seed=0).generate(60 * FS)
    noisy = SyntheticSource(seed=0, config=SyntheticConfig(muscle_rate_per_min=30)).generate(
        60 * FS
    )
    gain = band_power(noisy, 40, 100) / band_power(quiet, 40, 100)
    assert np.all(gain > 1.2), gain
    lateral = [channel(SyntheticSource(), label) for label in ("FC3", "FC4", "C3", "C4")]
    assert np.all(gain[lateral] > 3), gain


@pytest.mark.parametrize("count", [1, 2])
def test_jaw_clench_bursts_on_every_electrode(count: int) -> None:
    source = SyntheticSource(montage="C", seed=0)
    source.generate(2 * FS)
    source.clench(count)
    data = source.generate(2 * FS)
    high = sosfilt(butter(4, 40, "highpass", fs=FS, output="sos"), data, axis=0)
    # RMS over 50 ms, median over the electrodes: one value per sample.
    window = np.ones(FS // 20) / (FS // 20)
    rms = np.sqrt(np.convolve(np.median(high**2, axis=1), window, mode="same"))
    active = rms > 5 * np.median(rms[int(1.4 * FS) :])  # the bursts are over after 1.2 s
    rises = np.flatnonzero(np.diff(active.astype(int)) == 1) + 1
    # A burst edge can cross the threshold several times: group rises 0.25 s apart.
    onsets = [int(rises[0])] + [int(b) for a, b in pairwise(rises) if b - a > FS // 4]
    assert len(onsets) == count
    if count == 2:
        assert 0.5 * FS <= onsets[1] - onsets[0] <= 0.75 * FS
    # Every electrode sees the clench.
    gain = np.mean(high[:FS] ** 2, axis=0) / np.mean(high[int(1.4 * FS) :] ** 2, axis=0)
    assert np.all(gain > 5), gain
    markers = [m.payload for m in source.read_markers() if m.payload["type"] == "artifact"]
    assert markers == [
        {"type": "artifact", "label": "jaw_clench", "count": count, "source": "synthetic"}
    ]


def test_clench_rejects_other_counts() -> None:
    with pytest.raises(ValueError, match="1 or 2"):
        SyntheticSource().clench(3)


def test_line_noise_peak() -> None:
    config = SyntheticConfig(line_noise_uv=10, line_freq_hz=60)
    samples = SyntheticSource(seed=0, srate=500, config=config).generate(20 * 500)
    freqs, psd = welch(samples[:, 0], fs=500, nperseg=1000)
    assert abs(freqs[np.argmax(psd[freqs > 30]) + np.sum(freqs <= 30)] - 60) < 1


# --- Markers --------------------------------------------------------------------


def test_state_change_emits_marker_at_first_sample() -> None:
    source = SyntheticSource(seed=0)
    source.generate(100)
    source.set_state("left")
    source.set_eyes("closed")
    assert source.read_markers() == []  # nothing until the samples exist
    source.generate(10)
    markers = source.read_markers()
    assert [m.payload["type"] for m in markers] == ["state", "eyes"]
    assert markers[0].payload == {"type": "state", "label": "left", "source": "synthetic"}
    assert markers[0].timestamp == pytest.approx(100 / FS)
    assert markers[0].to_json() == '{"label":"left","source":"synthetic","type":"state"}'
    assert source.read_markers() == []  # markers are returned once


def test_setting_same_state_emits_nothing() -> None:
    source = SyntheticSource(seed=0)
    source.generate(10)
    source.set_state("rest")
    source.generate(10)
    assert source.read_markers() == []


def test_markers_can_be_disabled() -> None:
    source = SyntheticSource(seed=0, emit_markers=False)
    source.generate(10)
    source.set_state("right")
    source.blink()
    source.generate(10)
    assert source.read_markers() == []


# --- Real-time reading ----------------------------------------------------------


def test_read_is_paced_by_the_clock() -> None:
    clock = FakeClock(now=100.0)
    with SyntheticSource(seed=0, clock=clock) as source:
        samples, _ = source.read(max_samples=1000)
        assert samples.shape[0] == 0
        clock.now += 0.2  # 50 samples are due
        samples, timestamps = source.read(max_samples=1000)
        assert samples.shape == (50, 8)
        np.testing.assert_allclose(timestamps, 100.0 + np.arange(50) / FS)
        clock.now += 1.0  # 250 due, read by blocks of 100
        sizes = [source.read(max_samples=100)[0].shape[0] for _ in range(4)]
        assert sizes == [100, 100, 50, 0]


def test_read_matches_offline_generation() -> None:
    clock = FakeClock()
    with SyntheticSource(seed=5, clock=clock) as source:
        clock.now += 2.0
        online = np.vstack([source.read(max_samples=64)[0] for _ in range(8)])
    offline = SyntheticSource(seed=5).generate(online.shape[0])
    np.testing.assert_array_equal(online, offline)


def test_marker_timestamps_use_the_clock() -> None:
    clock = FakeClock(now=50.0)
    with SyntheticSource(seed=0, clock=clock) as source:
        clock.now += 0.4
        source.read(max_samples=1000)  # 100 samples
        source.set_state("feet")
        clock.now += 0.4
        _, timestamps = source.read(max_samples=1000)
        (marker,) = source.read_markers()
    assert marker.timestamp == pytest.approx(timestamps[0])
    assert marker.timestamp == pytest.approx(50.4)


def test_restart_continues_the_timeline() -> None:
    clock = FakeClock(now=10.0)
    source = SyntheticSource(seed=0, clock=clock)
    with source:
        clock.now += 1.0
        source.read(max_samples=1000)
    clock.now += 60.0  # paused for a minute
    with source:
        assert source.read(max_samples=1000)[0].shape[0] == 0
        clock.now += 0.1
        _, timestamps = source.read(max_samples=1000)
    assert timestamps[0] == pytest.approx(clock.now - 0.1)
    assert source.n_produced == 275


# --- Keyboard -------------------------------------------------------------------


def test_handle_key_maps_bindings() -> None:
    source = SyntheticSource(seed=0)
    assert set(KEY_BINDINGS) == {"r", "a", "d", "s", "e", "b", "j"}
    for key, state in zip("adsr", ("left", "right", "feet", "rest"), strict=True):
        assert source.handle_key(key)
        assert source.state == state
    assert source.handle_key("E")
    assert source.eyes == "closed"
    assert source.handle_key("e")
    assert source.eyes == "open"
    assert source.handle_key("b")
    assert source.handle_key("j")
    assert not source.handle_key("x")
    assert set(STATES) == {"rest", "left", "right", "feet"}
