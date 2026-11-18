"""Record 10 s of synthetic EEG to a file, then replay it on LSL.

The file goes to $OPENMIND_DATA_DIR/recordings/P00/ (default ~/openmind-data),
never into the repository.

    uv run python examples/record_and_replay.py
"""

from __future__ import annotations

from openmind_plug.outlet import stream
from openmind_plug.recorder import record_source
from openmind_plug.sources.replay import ReplaySource
from openmind_plug.sources.synthetic import SyntheticSource


def main() -> None:
    source = SyntheticSource(seed=0)
    source.set_state("left")
    print("Recording 10 s of synthetic EEG...")
    recorder = record_source(source, duration=10, subject="P00")
    path = recorder.save()
    print(f"Saved {recorder.n_samples} samples and {recorder.n_markers} markers to {path}")

    replay = ReplaySource.from_file(path, speed=2.0)
    print(f"Replaying at 2x on LSL ({replay.duration:.0f} s of data)...")
    stats = stream(replay)
    print(f"Replayed {stats.n_samples} samples in {stats.elapsed:.1f} s.")


if __name__ == "__main__":
    main()
