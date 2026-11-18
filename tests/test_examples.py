"""Smoke tests: the examples run against each other over LSL."""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.py")), ids=lambda p: p.name)
def test_examples_compile(path: Path) -> None:
    py_compile.compile(str(path), doraise=True)


def run(script: str, *args: str, timeout: float = 30) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(EXAMPLES / script), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def test_stream_synthetic_feeds_print_band_power() -> None:
    producer = run("stream_synthetic.py", "--duration", "6", "--period", "1")
    consumer = run("print_band_power.py", "--duration", "3")
    consumer_out, _ = consumer.communicate(timeout=30)
    producer_out, _ = producer.communicate(timeout=30)
    assert consumer.returncode == 0, consumer_out
    assert producer.returncode == 0, producer_out
    assert "state: left" in producer_out
    assert "Sent" in producer_out
    bars = [line for line in consumer_out.splitlines() if " C3 " in f" {line}"]
    assert len(bars) >= 2, consumer_out


@pytest.mark.slow
def test_record_and_replay(tmp_path: Path) -> None:
    process = subprocess.run(
        [sys.executable, str(EXAMPLES / "record_and_replay.py")],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "OPENMIND_DATA_DIR": str(tmp_path)},
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "Replayed 2500 samples" in process.stdout
    assert list((tmp_path / "recordings" / "P00").glob("P00_*_raw.fif"))
