"""Tests for the per-subject variability of the synthetic source."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from scipy.signal import welch

from openmind_plug.sources.synthetic import (
    ALPHA_HZ_RANGE,
    BETA_HZ_RANGE,
    CHANNEL_GAIN_RANGE,
    ERD_DEPTH_RANGE,
    ERD_DEPTH_WEAK_RANGE,
    SyntheticSource,
    SyntheticSubject,
)

FS = 250


def record(
    seconds: float = 20.0,
    state: str = "rest",
    eyes: str = "open",
    **kwargs: object,
) -> tuple[npt.NDArray[np.float32], SyntheticSource]:
    source = SyntheticSource(srate=FS, seed=5, **kwargs)  # type: ignore[arg-type]
    source.set_state(state)
    source.set_eyes(eyes)
    source.generate(FS)  # let the ERD settle
    return source.generate(int(seconds * FS)), source


def band(x: npt.NDArray[np.float32], ch: int, lo: float, hi: float) -> float:
    freqs, psd = welch(x[:, ch], fs=FS, nperseg=512)
    return float(psd[(freqs >= lo) & (freqs <= hi)].sum())


def test_default_subject_changes_nothing() -> None:
    plain, _ = record(state="left")
    explicit, _ = record(state="left", subject=SyntheticSubject.default())
    np.testing.assert_array_equal(plain, explicit)


def test_same_subject_and_seed_are_identical() -> None:
    a, _ = record(5.0, subject=3)
    b, _ = record(5.0, subject=3)
    np.testing.assert_array_equal(a, b)
    assert SyntheticSubject.sample(3, seed=1) == SyntheticSubject.sample(3, seed=1)
    assert SyntheticSubject.sample(3, seed=1) != SyntheticSubject.sample(3, seed=2)


def test_sampled_parameters_are_within_documented_ranges() -> None:
    weak = 0
    for subject_id in range(200):
        s = SyntheticSubject.sample(subject_id)
        assert ALPHA_HZ_RANGE[0] <= s.alpha_hz <= ALPHA_HZ_RANGE[1]
        assert ALPHA_HZ_RANGE[0] <= s.mu_hz <= ALPHA_HZ_RANGE[1]
        assert BETA_HZ_RANGE[0] <= s.beta_hz <= BETA_HZ_RANGE[1]
        assert s.erd_depth is not None and s.channel_gain is not None
        depths = list(s.erd_depth.values())
        if max(depths) <= ERD_DEPTH_WEAK_RANGE[1]:
            weak += 1
        else:
            assert all(ERD_DEPTH_RANGE[0] <= d <= ERD_DEPTH_RANGE[1] for d in depths)
        assert all(CHANNEL_GAIN_RANGE[0] <= g <= CHANNEL_GAIN_RANGE[1] for g in s.channel_gain)
    assert 5 <= weak <= 60


def test_subjects_differ_in_covariance_and_alpha_peak() -> None:
    low = SyntheticSubject(alpha_hz=8.8, mu_hz=8.8)
    high = SyntheticSubject(alpha_hz=11.5, mu_hz=11.5)
    a, source = record(30.0, eyes="closed", subject=low)
    b, _ = record(30.0, eyes="closed", subject=high)
    o1 = source.info.labels.index("O1") if "O1" in source.info.labels else 0
    freqs, pa = welch(a[:, o1], fs=FS, nperseg=1024)
    _, pb = welch(b[:, o1], fs=FS, nperseg=1024)
    band_mask = (freqs >= 7.5) & (freqs <= 13)
    assert abs(freqs[band_mask][np.argmax(pa[band_mask])] - 8.8) < 1.0
    assert abs(freqs[band_mask][np.argmax(pb[band_mask])] - 11.5) < 1.0

    cov1 = np.cov(record(20.0, subject=1)[0].T)
    cov2 = np.cov(record(20.0, subject=2)[0].T)
    assert np.linalg.norm(cov1 - cov2) / np.linalg.norm(cov1) > 0.05


def erd_ratio(subject: SyntheticSubject) -> float:
    """Mu+beta power on C4 relative to rest when imagining a left movement."""
    rest, source = record(40.0, "rest", subject=subject)
    left, _ = record(40.0, "left", subject=subject)
    c4 = source.info.labels.index("C4")
    return band(left, c4, 8, 30) / band(rest, c4, 8, 30)


def test_erd_depth_is_measurable_on_c4() -> None:
    depth = {"left": 0.7, "right": 0.7, "feet": 0.7}
    weak = {"left": 0.1, "right": 0.1, "feet": 0.1}
    strong_ratio = erd_ratio(SyntheticSubject(erd_depth=depth))
    weak_ratio = erd_ratio(SyntheticSubject(erd_depth=weak))
    assert strong_ratio < 0.65 < 0.85 < weak_ratio


def test_session_drift_is_small_but_nonzero() -> None:
    base = SyntheticSubject.sample(4)
    s1, s2 = base.with_session(1), base.with_session(2)
    assert s1 != s2 and s1 == base.with_session(1)
    assert base.channel_gain is not None and s1.channel_gain is not None
    ratio = np.array(s1.channel_gain) / np.array(base.channel_gain)
    assert np.all(np.abs(ratio - 1) <= 0.05 + 1e-9) and np.any(ratio != 1)
    a, _ = record(5.0, subject=4)
    b, _ = record(5.0, subject=4, session=1)
    assert not np.array_equal(a, b)
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 0.15


def test_session_without_subject_uses_neutral_subject() -> None:
    a, _ = record(2.0)
    b, _ = record(2.0, session=1)
    assert not np.array_equal(a, b)


def test_invalid_subjects_are_rejected() -> None:
    with pytest.raises(ValueError, match="subject_id"):
        SyntheticSubject(subject_id=-1)
    with pytest.raises(ValueError, match="erd_depth"):
        SyntheticSubject(erd_depth={"left": 0.5})
    with pytest.raises(ValueError, match="square"):
        SyntheticSubject(spatial_mixing=((1.0, 0.0),))
    with pytest.raises(ValueError, match="channels"):
        SyntheticSource(montage="A", subject=SyntheticSubject(channel_gain=(1.0,)))
