"""Stream the synthetic source and cycle through motor-imagery states.

Every few seconds the simulated user switches between rest, left, rest, right,
with a cue marker at each change: a ready-made calibration session for testing a
decoder without a headset.

    uv run python examples/stream_synthetic.py --duration 60
"""

from __future__ import annotations

import argparse
import itertools
import threading

from openmind_plug.outlet import stream
from openmind_plug.sources.synthetic import SyntheticConfig, SyntheticSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--duration", type=float, default=None, help="seconds (default: forever)")
    parser.add_argument("--period", type=float, default=4.0, help="seconds per state")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = SyntheticSource(seed=args.seed, config=SyntheticConfig(blink_rate_per_min=10))
    stop = threading.Event()

    def cycle_states() -> None:
        for state in itertools.cycle(["rest", "left", "rest", "right"]):
            source.set_state(state)
            print(f"state: {state}", flush=True)
            if stop.wait(args.period):
                return

    threading.Thread(target=cycle_states, daemon=True).start()
    print("Streaming openmind-eeg; press Ctrl+C to stop.", flush=True)
    stats = stream(source, duration=args.duration, stop_event=stop)
    stop.set()
    print(f"Sent {stats.n_samples} samples and {stats.n_markers} markers.")


if __name__ == "__main__":
    main()
