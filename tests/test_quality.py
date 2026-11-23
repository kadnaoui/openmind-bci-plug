"""Tests for signal-quality checks, the alpha test and the quality command."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest
from typer.testing import CliRunner

from openmind_plug.cli import app
from openmind_plug.outlet import EEGOutlet, stream
from openmind_plug.quality import alpha_test, assess, band_power
from openmind_plug.sources.synthetic import SyntheticConfig, SyntheticSource

FS = 250.0
runner = CliRunner()


def synthetic(seconds: float = 4.0, montage: str = "A", **config: float) -> np.ndarray:
    source = SyntheticSource(montage=montage, seed=0, config=SyntheticConfig(**config))
    return source.generate(int(seconds * FS)).astype(np.float64)


LABELS_A = SyntheticSource(montage="A").info.labels


def statuses(samples: np.ndarray, **kwargs: object) -> dict[str, str]:
    report = assess(samples, FS, LABELS_A, **kwargs)  # type: ignore[arg-type]
    return {channel.label: channel.status for channel in report}


# --- Per-channel checks -----------------------------------------------------------


def test_clean_synthetic_signal_is_good() -> None:
    report = assess(synthetic(), FS, LABELS_A)
    assert [c.status for c in report] == ["good"] * 8
    assert all(c.ok and 5 < c.std_uv < 40 for c in report)
    assert all(c.impedance_kohm is None for c in report)


def test_zeroed_channel_is_flat() -> None:
    data = synthetic()
    data[:, 2] = 0.0
    assert statuses(data)["C3"] == "flat"
    assert list(statuses(data).values()).count("good") == 7


def test_constant_offset_is_flat_not_saturated() -> None:
    data = synthetic()
    data[:, 0] = 3000.0  # disconnected input stuck at a DC level
    assert statuses(data)["FC3"] == "flat"


def test_clipped_channel_is_saturated() -> None:
    data = synthetic()
    data[:, 4] = np.clip(data[:, 4] * 10, -60, 60)
    assert statuses(data)["C4"] == "saturated"


def test_saturation_threshold_when_range_is_known() -> None:
    data = synthetic()
    data[::5, 3] = 5000.0  # 20 % of samples at the rail
    assert statuses(data, saturation_uv=4000.0)["Cz"] == "saturated"


def test_line_noise_is_detected_on_one_channel() -> None:
    data = synthetic()
    t = np.arange(data.shape[0]) / FS
    data[:, 5] += 30 * np.sin(2 * np.pi * 50 * t)
    report = {c.label: c for c in assess(data, FS, LABELS_A)}
    assert report["CP3"].status == "line noise"
    assert report["CP3"].line_noise_ratio > 10
    assert report["C3"].line_noise_ratio < 3


def test_line_frequency_is_configurable() -> None:
    data = synthetic()
    t = np.arange(data.shape[0]) / FS
    data[:, 5] += 30 * np.sin(2 * np.pi * 60 * t)
    assert statuses(data)["CP3"] == "good"  # looking at 50 Hz
    assert statuses(data, line_freq=60.0)["CP3"] == "line noise"


def test_noisy_and_weak_channels() -> None:
    data = synthetic()
    data[:, 6] *= 10
    data[:, 1] /= 10
    result = statuses(data)
    assert result["CP4"] == "noisy"
    assert result["FC4"] == "weak"


def test_dc_offset_and_drift_do_not_hide_good_channels() -> None:
    data = synthetic()
    data += 20_000 + np.linspace(0, 500, data.shape[0])[:, None]
    assert set(statuses(data).values()) == {"good"}


def test_impedances_are_reported() -> None:
    report = assess(synthetic(), FS, LABELS_A, impedances_kohm={"C3": 12.5})
    assert {c.label: c.impedance_kohm for c in report}["C3"] == 12.5


def test_assess_validates_input() -> None:
    with pytest.raises(ValueError, match="shape"):
        assess(np.zeros((500, 3)), FS, LABELS_A)
    with pytest.raises(ValueError, match="1 s"):
        assess(np.zeros((100, 8)), FS, LABELS_A)


# --- Alpha test -----------------------------------------------------------------


def test_alpha_test_detects_eyes_closed_on_synthetic_source() -> None:
    source = SyntheticSource(montage="B", seed=0)
    eyes_open = source.generate(int(30 * FS))
    source.set_eyes("closed")
    source.generate(int(FS))
    eyes_closed = source.generate(int(30 * FS))
    report = alpha_test(eyes_open, eyes_closed, FS, source.info.labels)
    assert report.passed
    assert report.best_channel in {"O1", "O2"}
    assert report.best_gain > 4
    assert report.gains["Cz"] < 1.5


def test_alpha_test_fails_without_change() -> None:
    source = SyntheticSource(montage="B", seed=0)
    first = source.generate(int(20 * FS))
    second = source.generate(int(20 * FS))
    report = alpha_test(first, second, FS, source.info.labels)
    assert not report.passed
    assert report.best_gain < 1.5


def test_band_power_finds_a_sinusoid() -> None:
    t = np.arange(int(10 * FS)) / FS
    data = np.column_stack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 25 * t)])
    power = band_power(data, FS, (8, 12))
    assert power[0] > 100 * power[1]


# --- Command line ---------------------------------------------------------------


@contextmanager
def streaming(source: SyntheticSource) -> Iterator[str]:
    source_id = f"openmind-plug-test:{uuid.uuid4()}"
    stop = threading.Event()
    outlet = EEGOutlet(source.info, source_id=source_id)
    thread = threading.Thread(
        target=stream, args=(source,), kwargs={"stop_event": stop, "outlet": outlet}
    )
    thread.start()
    try:
        yield source_id
    finally:
        stop.set()
        thread.join(timeout=5)
        outlet.close()


def test_cli_quality_live() -> None:
    with streaming(SyntheticSource(seed=0)) as source_id:
        result = runner.invoke(
            app, ["quality", "--source-id", source_id, "--window", "1", "--duration", "2"]
        )
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.startswith("[")]
    assert len(lines) == 2
    assert all(line.count("good") == 8 for line in lines)


def close_eyes_on_second_phase(
    monkeypatch: pytest.MonkeyPatch, source: SyntheticSource
) -> list[float]:
    """Make the CLI close the source's eyes exactly when the second phase starts."""
    import openmind_plug.cli as cli

    original = cli._collect
    calls: list[float] = []

    def collect(inlet: object, seconds: float) -> object:
        calls.append(seconds)
        if len(calls) == 2:
            source.set_eyes("closed")
        return original(inlet, seconds)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "_collect", collect)
    return calls


def test_cli_alpha_test_passes_when_eyes_close(monkeypatch: pytest.MonkeyPatch) -> None:
    source = SyntheticSource(montage="B", seed=0)
    calls = close_eyes_on_second_phase(monkeypatch, source)
    with streaming(source) as source_id:
        result = runner.invoke(
            app, ["quality", "--source-id", source_id, "--alpha-test", "--phase", "3"]
        )
    assert result.exit_code == 0, result.output
    assert "Keep your eyes OPEN" in result.output
    assert "unreliable" in result.output
    assert "PASSED" in result.output
    assert calls == [3.0, 3.0]


def test_cli_alpha_test_fails_when_alpha_does_not_rise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Inverted subject: alpha drops when the eyes close, so the test must fail.
    config = SyntheticConfig(alpha_open_uv=20, alpha_closed_uv=2)
    source = SyntheticSource(montage="B", seed=0, config=config)
    close_eyes_on_second_phase(monkeypatch, source)
    with streaming(source) as source_id:
        result = runner.invoke(
            app, ["quality", "--source-id", source_id, "--alpha-test", "--phase", "3"]
        )
    assert result.exit_code == 1, result.output
    assert "FAILED" in result.output


def test_cli_quality_stream_not_found() -> None:
    result = runner.invoke(app, ["quality", "--source-id", "nope", "--timeout", "0.3"])
    assert result.exit_code == 2
