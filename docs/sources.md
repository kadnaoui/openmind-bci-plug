# Sources

A **source** is anything that produces EEG samples: a headset driver, a replayed
dataset, a synthetic generator or a relayed LSL stream. Every source implements
the same small interface, so the LSL output, the recorder and the quality checks
work with all of them.

## Stream metadata

### `ChannelInfo`

| Field | Default | Rule |
|---|---|---|
| `label` | (required) | Non-empty, no surrounding whitespace. Prefer 10-20/10-10 labels (`C3`, `Cz`). |
| `unit` | `"uV"` | One of `uV`, `mV`, `V`. The standard stream uses `uV`. |
| `type` | `"EEG"` | One of `EEG`, `EOG`, `EMG`, `ECG`, `MISC`. |

### `StreamInfo`

| Field | Default | Rule |
|---|---|---|
| `name` | `"openmind-eeg"` | Non-empty. |
| `srate` | (required) | Nominal sampling rate in Hz, finite and > 0. |
| `channels` | (required) | At least one; labels unique, case-insensitively (`Cz` and `CZ` clash). |
| `manufacturer` | `"unknown"` | Hardware maker, or `OpenMind` for software sources. |
| `model` | `"unknown"` | Device model or source kind (`IronBCI-8`, `synthetic`...). |
| `driver` | `"unknown"` | Plug driver that produced the stream. |
| `reference` | `"unknown"` | Reference electrode or scheme (`Cz`, `average`...). |
| `montage` | `None` | Montage name, if a named montage is used. |

Both are immutable [pydantic](https://docs.pydantic.dev) models: invalid values
raise `pydantic.ValidationError` at construction.

```python
from openmind_plug.montage import get_montage
from openmind_plug.sources import StreamInfo

montage = get_montage("A")
info = StreamInfo(
    srate=250.0,
    channels=montage.channels(),
    manufacturer="OpenMind",
    model="synthetic",
    driver="synthetic",
    reference="Cz",
    montage=montage.name,
)
info.labels  # ('FC3', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CP4', 'Fp1')
info.n_channels  # 8
```

## The `EEGSource` interface

```python
with source:  # start() ... stop()
    samples, timestamps = source.read(max_samples=64)
```

- `info` → the `StreamInfo` of the source.
- `start()` / `stop()` → open and release the device; both are idempotent. The
  `with` statement calls them for you, and `stop()` runs even if an error occurs.
- `read(max_samples)` → the samples available now, without blocking for long:
  - `samples`: `float32` array of shape `(n_samples, n_channels)`, in the channels' units,
    with `0 <= n_samples <= max_samples`;
  - `timestamps`: `float64` array of shape `(n_samples,)`, in seconds on the
    **LSL clock** (`pylsl.local_clock()`), never `time.time()`.

Reading before `start()`, or a driver returning arrays of the wrong shape, raises
`SourceStateError`.

### Writing a new source

Subclass `EEGSource` and implement three hooks; the base class handles state
checks and validates what you return.

```python
import numpy as np
from openmind_plug.sources import EEGSource, StreamInfo


class MySource(EEGSource):
    def __init__(self, info: StreamInfo) -> None:
        super().__init__()
        self._info = info

    @property
    def info(self) -> StreamInfo:
        return self._info

    def _start(self) -> None:
        # Open the device.
        ...

    def _stop(self) -> None:
        # Close the device.
        ...

    def _read(self, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
        # Return (samples, timestamps).
        ...
```

## Montages

A montage is the ordered list of electrodes wired to the amplifier inputs.

| Name | Electrodes (input order) | Use |
|---|---|---|
| **`A`** (default) | `FC3, FC4, C3, Cz, C4, CP3, CP4, Fp1` | Motor imagery: sensorimotor cortex, plus `Fp1` for blink detection |
| `B` | `C3, Cz, C4, CP3, CP4, Fp1, O1, O2` | SSVEP variant: occipital `O1`/`O2` instead of `FC3`/`FC4` |

```python
from openmind_plug.montage import get_montage, position

montage = get_montage("b")  # case-insensitive; unknown names list the available ones
montage.channels()  # tuple of ChannelInfo, unit "uV", type "EEG"
montage.positions()  # {"C3": (x, y, z), ...}
position("Cz")  # (0.0004, -0.0092, 0.1002)
```

Positions are approximate coordinates in meters for 47 electrodes of the 10-10
system, in the MNE head frame (x towards the right ear, y towards the nose,
z towards the vertex), taken from the MNE `standard_1005` template. They are meant
for visualization and spatial filtering, not for source localization.

## Synthetic source

`SyntheticSource` generates realistic EEG without any hardware. It is the main
test tool of the OpenMind ecosystem: deterministic with a seed, with a
controllable motor-imagery state and optional artifacts.

```python
from openmind_plug.sources.synthetic import SyntheticConfig, SyntheticSource

source = SyntheticSource(montage="A", srate=250, seed=42)
with source:
    source.set_state("left")  # imagine a left-hand movement
    samples, timestamps = source.read(max_samples=64)
    markers = source.read_markers()  # [Marker(timestamp, {"type": "state", ...})]
```

### Signal model

Each channel, in microvolts, is the sum of:

| Component | Model | Default |
|---|---|---|
| Background | 1/f (pink) noise, independent per channel | 15 µV RMS |
| Alpha (8-12 Hz) | Posterior generator, weight 0 at the central line → 1 at O1/O2 | 4 µV eyes open, 20 µV eyes closed |
| Mu (8-12 Hz) | Three generators centred on C3, Cz and C4 (Gaussian spread, σ = 2.5 cm) | 10 µV |
| Beta (13-30 Hz) | Same three sites | 4 µV |

Rhythm amplitudes fluctuate slowly (±30 %, time constant 0.5 s), as real
rhythms wax and wane. Amplitudes are peak values at the centre of each generator.

### State and event-related desynchronization

Imagining a movement lowers mu and beta power over the opposite hemisphere:

| State | Generators lowered | Key |
|---|---|---|
| `rest` | none | `r` |
| `left` | C4 (right hemisphere) | `a` |
| `right` | C3 (left hemisphere) | `d` |
| `feet` | Cz | `s` |

The power drop is `erd_depth` (default 60 %), reached smoothly with time
constant `erd_tau_s` (default 0.3 s). `set_eyes("closed")` (key `e` toggles)
raises posterior alpha the same way.

### Artifacts (all off by default)

| Artifact | Model | Settings |
|---|---|---|
| Blink | Hann pulse of 0.25-0.4 s, strongest on Fp1/Fp2 (σ = 3 cm) | `blink_rate_per_min`, `blink_uv` (120), or `blink()` / key `b` |
| Muscle | Burst of 0.5-1.5 s of noise above 30 Hz, stronger on lateral channels | `muscle_rate_per_min`, `muscle_uv` (20 µV RMS) |
| Power line | Sinusoid with a per-channel gain | `line_noise_uv`, `line_freq_hz` (50 or 60) |

```python
config = SyntheticConfig(blink_rate_per_min=15, line_noise_uv=5)
source = SyntheticSource(seed=0, config=config)
```

### Markers (ground truth)

With `emit_markers=True` (default), every state change, eyes change and artifact
produces a marker whose timestamp is that of the first EEG sample it applies to:

```json
{"type": "state", "label": "left", "source": "synthetic"}
{"type": "eyes", "label": "closed", "source": "synthetic"}
{"type": "artifact", "label": "blink", "source": "synthetic"}
```

### Timing and reproducibility

- `read()` returns the samples due according to the LSL clock since `start()`;
  timestamps are perfectly regular (`t0 + index / srate`). After `stop()` and
  `start()`, the stream resumes with no gap in sample indices.
- `generate(n)` returns the next `n` samples immediately, ignoring the clock —
  handy for offline tests and datasets.
- For a given seed and sequence of state changes, the signal only depends on the
  sample index, not on how it is split into reads.
- `srate` must be at least 100 Hz.

## Replay source

`ReplaySource` plays back a recording as if it were live: a public dataset
downloaded with MOABB, or any file MNE can read (FIF, EDF, BDF, GDF,
BrainVision...). Install the extra first:

```bash
pip install "openmind-plug[replay]"
```

```python
from openmind_plug.sources.replay import ReplaySource

# Public dataset (downloaded on first use)
source = ReplaySource.from_dataset("physionetmi", subject=1, montage="A")

# Any file readable by MNE
source = ReplaySource.from_file("session_raw.fif", montage="A", loop=True)

with source:
    samples, timestamps = source.read(max_samples=64)
    markers = source.read_markers()
```

### Supported datasets

| Key | Dataset | Channels | Rate | Classes | Fits montage |
|---|---|---|---|---|---|
| `bnci2014_001` | BCI Competition IV 2a, 9 subjects | 22 EEG (no Fp1, no occipital) | 250 Hz | left, right, feet, tongue | none: use `montage=None` |
| `physionetmi` | PhysioNet EEGMMIDB, 109 subjects | 64 EEG (10-10) | 160 Hz | left, right, hands, feet, rest | `A`, `B` |

`from_dataset(dataset, subject=1, session=None, runs=None, ...)` loads the first
session by default and concatenates all its runs, in order.

### Processing

1. **Channel selection.** With a montage, its electrodes are picked in montage
   order; channel names are normalized (`"Fc3."` → `FC3`, `"EEG CZ"` → `Cz`).
   If an electrode is missing, `MissingChannelsError` names it and lists the
   montages that fit the recording (or suggests `montage=None`). Without a
   montage (the default), every EEG channel is kept; EOG, EMG and stim channels
   are dropped.
2. **Units.** Data are converted from volts (MNE convention) to microvolts.
3. **Resampling** to `srate` (250 Hz by default) when the recording differs.
4. **Events → markers.** Annotations (or stim-channel codes, for datasets that
   use them) become cue markers; `BAD_*` and `EDGE_*` annotations are ignored.
   Dataset names are translated (`left_hand` → `left`, `right_hand` → `right`,
   ...):

   ```json
   {"type": "cue", "label": "left", "trial": 12, "source": "replay"}
   ```

   `trial` counts cues from 1 and keeps counting across loops.

   Annotations written by the recorder (`state/left`, `artifact/blink`...) are
   replayed with their own type and label, without `trial`. Recordings also
   restore their model, reference and montage (see [recording](recording.md)).

### Playback

- `read()` releases samples at the pace of the LSL clock; `speed=10.0` plays ten
  times faster (timestamps follow the accelerated clock, so EEG and markers stay
  aligned).
- Without `loop`, the stream ends with the recording (`finished` becomes true and
  `read()` returns empty blocks); with `loop=True`, it restarts seamlessly.
- After `stop()` and `start()`, playback resumes where it stopped.

### Where data go

Downloads are stored in `$OPENMIND_DATA_DIR/mne_data` (default
`~/openmind-data/mne_data`), never in the repository. Plug sets MNE's
`MNE_DATASETS_*_PATH` variables only while loading, so your global MNE
configuration is left untouched.
