"""Show mu-band power on C3 and C4 as text bars, from the live standard stream.

A minimal visualization with no extra dependency: run it next to
``stream_synthetic.py`` (or ``openmind-plug stream``) and watch the bar of the
hemisphere opposite to the imagined hand shrink.

    uv run python examples/print_band_power.py
"""

from __future__ import annotations

import argparse

import numpy as np

from openmind_plug.inlet import EEGInlet
from openmind_plug.quality import band_power


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=float, default=1.0, help="seconds per update")
    parser.add_argument("--duration", type=float, default=None, help="seconds (default: forever)")
    parser.add_argument("--channels", default="C3,C4", help="comma-separated labels")
    args = parser.parse_args()

    with EEGInlet(timeout=10) as inlet:
        labels = [label for label in args.channels.split(",") if label in inlet.info.labels]
        if not labels:
            raise SystemExit(f"none of {args.channels} in {inlet.info.labels}")
        columns = [inlet.info.labels.index(label) for label in labels]
        srate = inlet.info.srate
        n_window = int(args.window * srate)
        buffer = np.empty((0, inlet.info.n_channels), dtype=np.float32)
        elapsed = 0.0
        print(f"mu power (8-13 Hz), {args.window:g} s windows; Ctrl+C to stop")
        try:
            while args.duration is None or elapsed < args.duration:
                samples, _ = inlet.read(timeout=0.2)
                buffer = np.vstack([buffer, samples])
                if buffer.shape[0] < n_window:
                    continue
                window, buffer = buffer[:n_window], buffer[n_window:]
                elapsed += args.window
                power = band_power(window[:, columns], srate, (8.0, 13.0))
                bars = "  ".join(
                    f"{label:>3} {'#' * min(int(p), 40):<40}"
                    for label, p in zip(labels, power, strict=True)
                )
                markers = " ".join(m.payload.get("label", "") for m in inlet.read_markers())
                print(f"{bars}  {markers}", flush=True)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
