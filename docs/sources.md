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
