# Adding a headset driver

A driver is an `EEGSource` subclass. Implement it, and the LSL output, the
recorder, the quality checks and the CLI work with your headset for free.

## 1. Describe the stream

Build a `StreamInfo` once, in the constructor:

```python
from openmind_plug.montage import get_montage
from openmind_plug.sources import StreamInfo

montage = get_montage("A")  # or your own ChannelInfo tuple
info = StreamInfo(
    srate=250.0,
    channels=montage.channels(unit="uV"),  # the unit your samples are in
    manufacturer="Acme",
    model="Acme-8",
    driver="acme",
    reference="Cz",
    montage=montage.name,
)
```

- Use the unit of the values you return (`uV`, `mV` or `V`); the outlet converts
  to microvolts. Avoid converting yourself unless the hardware gives raw counts.
- Use 10-10 labels (`C3`, `Cz`) when electrodes have standard positions.

## 2. Implement three hooks

```python
import numpy as np
import pylsl

from openmind_plug.sources import EEGSource, SampleArray, StreamInfo, TimestampArray


class AcmeSource(EEGSource):
    def __init__(self, port: str) -> None:
        super().__init__()
        self._port = port
        self._info = ...  # StreamInfo as above

    @property
    def info(self) -> StreamInfo:
        return self._info

    def _start(self) -> None:
        # Open the device and start acquisition.
        ...

    def _stop(self) -> None:
        # Stop acquisition and release the device; must not raise if already closed.
        ...

    def _read(self, max_samples: int) -> tuple[SampleArray, TimestampArray]:
        # Return what arrived since the last call, without blocking for long.
        samples = ...  # (n, n_channels) float32, n <= max_samples
        timestamps = ...  # (n,) float64, LSL clock
        return samples, timestamps
```

The base class handles `start()` / `stop()` idempotence, the `with` statement and
checks every block you return (shape, dtype, `max_samples`).

## 3. Get timestamps right

- Use `pylsl.local_clock()`, never `time.time()`: other OpenMind components
  synchronize on the LSL clock.
- Stamp each sample with its **acquisition** time. If the device has a fixed rate,
  anchor once and count samples: `t0 + index / srate`. If packets arrive in
  bursts over Bluetooth, the stamp of the burst arrival is not the stamp of each
  sample.
- When the hardware numbers its packets, detect lost packets and **skip** the
  missing samples (do not invent values): consumers see the gap in timestamps and
  the recorder marks it `BAD_gap`.

## 4. Optional extras

- `read_markers()`: return events the device reports (button, trigger input).
- `impedances()`: return `{label: kΩ}` if the hardware measures impedance; the
  quality report shows it.
- Put heavy or platform-specific dependencies in an optional extra in
  `pyproject.toml`, and import them inside the driver module.

## 5. Register and test

1. Add the source to `SourceName` and `SOURCES` in `cli.py`, and its options to
   `stream`.
2. Test without hardware: replay recorded packets or simulate the device (see
   `tests/test_lsl_relay.py` for a simulated stream). Test with and without the
   optional extra installed.
3. Validate a live stream: `openmind-plug check-stream`, then
   `openmind-plug quality --alpha-test` with a real person.
4. Document the driver in `docs/sources.md` (and a checklist under
   `docs/hardware/` for hardware validation).
