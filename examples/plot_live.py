"""Plot the live standard stream with matplotlib (scrolling traces).

matplotlib comes with MNE, a dependency of plug:

    uv run python examples/plot_live.py
"""

from __future__ import annotations

import argparse

import numpy as np

from openmind_plug.inlet import EEGInlet


def main() -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.artist import Artist
    except ImportError as error:
        raise SystemExit("this example needs matplotlib: pip install matplotlib") from error

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=5.0, help="time span shown")
    parser.add_argument("--spacing", type=float, default=100.0, help="µV between traces")
    args = parser.parse_args()

    inlet = EEGInlet(timeout=10)
    info = inlet.info
    n_shown = int(args.seconds * info.srate)
    data = np.zeros((n_shown, info.n_channels), dtype=np.float32)
    times = np.arange(n_shown) / info.srate - args.seconds
    offsets = -args.spacing * np.arange(info.n_channels)

    figure, axis = plt.subplots(figsize=(10, 6))
    lines = axis.plot(times, data + offsets, linewidth=0.8)
    axis.set_yticks(offsets, info.labels)
    axis.set_xlabel("time (s)")
    axis.set_ylim(offsets[-1] - args.spacing, args.spacing)
    axis.set_title(f"{inlet.source_id} ({info.srate:g} Hz, µV)")

    def update(_frame: int) -> list[Artist]:
        nonlocal data
        samples, _ = inlet.read()
        if samples.shape[0]:
            data = np.vstack([data, samples])[-n_shown:]
            centred = data - data.mean(axis=0)
            for i, line in enumerate(lines):
                line.set_ydata(centred[:, i] + offsets[i])
        return list(lines)

    _animation = FuncAnimation(figure, update, interval=50, cache_frame_data=False)
    plt.show()
    inlet.close()


if __name__ == "__main__":
    main()
