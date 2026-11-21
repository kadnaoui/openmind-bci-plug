# Getting started without a headset (5 minutes)

This guide streams realistic synthetic EEG on your machine, checks it, and
watches the motor-imagery effect live. No hardware, no download.

## 1. Install (1 min)

```bash
git clone https://github.com/openmind-bci/plug.git
cd plug
uv sync                     # or: pip install openmind-plug
uv run openmind-plug --version
```

## 2. Stream synthetic EEG (1 min)

```bash
uv run openmind-plug stream --source synthetic --montage A
```

You now publish `openmind-eeg`: 8 channels (FC3, FC4, C3, Cz, C4, CP3, CP4, Fp1)
at 250 Hz, in microvolts, plus an `openmind-markers` stream. Leave it running.
(Other programs, such as the calibration scenes of OpenMind Sandbox, may publish
`openmind-markers` streams too: consumers read and merge all of them.)
In this terminal, single keys change what the simulated user does:

| Key | Effect |
|---|---|
| `a` / `d` | imagine the **left** / **right** hand: mu rhythm drops on C4 / C3 |
| `s` | imagine the **feet**: drop on Cz |
| `r` | back to **rest** |
| `e` | close / open the eyes: alpha rises at the back of the head |
| `b` | blink: large slow wave on Fp1 |
| `j` | double jaw clench: two short muscle bursts on every electrode |
| `q` | quit |

## 3. Check the stream (1 min)

In a second terminal:

```bash
uv run openmind-plug check-stream      # metadata, rate, timestamps: should print OK
uv run openmind-plug quality           # every channel should be "good"
```

## 4. See the effect (2 min)

```bash
uv run python examples/print_band_power.py
```

Each line shows mu-band power on C3 and C4. Press `a` in the first terminal:
after a fraction of a second the **C4** bar shrinks (desynchronization over the
hemisphere opposite to the imagined hand). Press `r`, then `d`: now **C3**
shrinks. For scrolling traces instead:

```bash
uv run python examples/plot_live.py
```

## Next steps

- Replay real EEG from a public dataset:
  `uv sync --extra replay`, then
  `uv run openmind-plug stream --source replay --dataset physionetmi --montage A`.
- Record a session: `uv run openmind-plug record --subject P01 --duration 60`
  ([recording](recording.md)).
- Feed any LSL consumer: OpenMind core, LabRecorder, MNE-LSL. The stream format is
  in the [specification](stream-spec.md).
- Scripted calibration session: `uv run python examples/stream_synthetic.py`.
