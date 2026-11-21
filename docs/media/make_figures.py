"""Draw the README figures of the synthetic source, in light and dark themes.

    uv run python docs/media/make_figures.py

Writes ``synthetic-eeg-{light,dark}.png`` and ``synthetic-spectra-{light,dark}.png``
next to this script. The source is seeded, so the figures are reproducible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy import signal

from openmind_plug.sources.synthetic import SyntheticSource

OUT = Path(__file__).parent
FS = 250.0
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "bg": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "grid": "#e4e3df",
        "band": "#f1f0ec",
        "rest": "#2a78d6",
        "left": "#eb6834",
        "right": "#1baf7a",
    },
    "dark": {
        "bg": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "grid": "#34332f",
        "band": "#252523",
        "rest": "#3987e5",
        "left": "#d95926",
        "right": "#199e70",
    },
}
PLAN = [("rest", 3), ("left", 4), ("rest", 3), ("right", 4), ("rest", 2), ("action", 2)]


def traces() -> tuple[np.ndarray, list[str], list[tuple[float, float, str]]]:
    """Montage C through rest, left, right and a double jaw clench."""
    source = SyntheticSource(montage="C", srate=FS, seed=7, emit_markers=False)
    chunks, spans, start = [], [], 0.0
    for state, seconds in PLAN:
        if state == "action":
            source.set_state("rest")
            source.clench(2)
        else:
            source.set_state(state)
        chunks.append(source.generate(int(seconds * FS)))
        spans.append((start, start + seconds, state))
        start += seconds
    labels = [channel.label for channel in source.info.channels]
    return np.concatenate(chunks), labels, spans


def spectra() -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Welch spectra per state; same seed, so only the state differs."""
    result = {}
    for state in ("rest", "left", "right"):
        source = SyntheticSource(montage="C", srate=FS, seed=3, emit_markers=False)
        source.set_state(state)
        source.generate(int(2 * FS))  # let the desynchronization settle
        x = source.generate(int(60 * FS))
        freqs, power = signal.welch(x, fs=FS, nperseg=int(2 * FS), axis=0)
        labels = [channel.label for channel in source.info.channels]
        db = {label: 10 * np.log10(power[:, i]) for i, label in enumerate(labels)}
        result[state] = (freqs, db)
    return result


def style(ax: Axes, theme: dict[str, str]) -> None:
    ax.set_facecolor(theme["bg"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=theme["ink2"], length=0, labelsize=10)


def draw_traces(theme_name: str, theme: dict[str, str]) -> None:
    x, labels, spans = traces()
    x = signal.sosfiltfilt(signal.butter(2, [1, 40], "bandpass", fs=FS, output="sos"), x, axis=0)
    t = np.arange(len(x)) / FS
    step = 60.0
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=110, facecolor=theme["bg"])
    style(ax, theme)
    for i in range(len(labels)):
        ax.plot(t, x[:, i] - i * step, color=theme["ink"], lw=0.7)
    for start, stop, state in spans:
        if state != "rest":
            ax.axvspan(start, stop, color=theme["band"], zorder=0, lw=0)
        ax.text(
            (start + stop) / 2,
            step * 0.9,
            state,
            ha="center",
            va="bottom",
            color=theme["ink2"],
            fontsize=11,
        )
    ax.set_yticks([-i * step for i in range(len(labels))], labels)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-(len(labels) - 0.3) * step, step * 1.5)
    ax.set_xlabel("time (s)", color=theme["ink2"])
    ax.set_title(
        "Synthetic EEG, montage C (1-40 Hz, 60 µV between channels)",
        color=theme["ink"],
        loc="left",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / f"synthetic-eeg-{theme_name}.png", facecolor=theme["bg"])
    plt.close(fig)


def draw_spectra(theme_name: str, theme: dict[str, str]) -> None:
    data = spectra()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=110, facecolor=theme["bg"], sharey=True)
    panels = [("C3", "right", "C3 (left hemisphere)"), ("C4", "left", "C4 (right hemisphere)")]
    for ax, (channel, move, title) in zip(axes, panels, strict=True):
        style(ax, theme)
        ax.grid(axis="y", color=theme["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for state, label in (("rest", "rest"), (move, f"{move} hand imagined")):
            freqs, power = data[state]
            keep = (freqs >= 4) & (freqs <= 35)
            ax.plot(freqs[keep], power[channel][keep], color=theme[state], lw=2, label=label)
        ax.axvspan(8, 13, color=theme["band"], zorder=0, lw=0)
        ax.set_title(title, color=theme["ink"], loc="left", fontsize=12)
        ax.set_xlabel("frequency (Hz)", color=theme["ink2"])
        ax.legend(frameon=False, labelcolor=theme["ink"], fontsize=10, loc="upper right")
    axes[0].set_ylabel("power (dB µV²/Hz)", color=theme["ink2"])
    title = "Imagining a hand movement lowers mu (8-13 Hz) and beta power on the opposite side"
    fig.suptitle(title, color=theme["ink"], x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"synthetic-spectra-{theme_name}.png", facecolor=theme["bg"])
    plt.close(fig)


def main() -> None:
    for name, theme in THEMES.items():
        draw_traces(name, theme)
        draw_spectra(name, theme)


if __name__ == "__main__":
    main()
