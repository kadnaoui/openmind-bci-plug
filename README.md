# OpenMind Plug

[![CI](https://github.com/openmind-bci/plug/actions/workflows/ci.yml/badge.svg)](https://github.com/openmind-bci/plug/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**The universal EEG plug.** Read any EEG source (a real headset, a recorded file,
a public dataset or a synthetic signal) and stream it as a **standard
[Lab Streaming Layer](https://labstreaminglayer.org) (LSL) stream** with complete metadata.

> [!WARNING]
> **Research and experimentation project — not a medical device.**
> OpenMind Plug must not be used for diagnosis, treatment or any clinical decision.

Part of the [OpenMind BCI](https://github.com/openmind-bci) platform:

| Repository | Role |
|---|---|
| **plug** (this one) | EEG source → standard LSL stream |
| [core](https://github.com/openmind-bci/core) | LSL stream → user intent → MQTT commands |
| [sandbox](https://github.com/openmind-bci/sandbox) | Virtual and real objects driven by those commands |

Plug does no AI and no decoding: it is useful to anyone who needs a clean,
well-described EEG stream, inside or outside OpenMind.

## Status

Early development (pre-alpha). The package skeleton is in place; sources,
LSL output and the CLI are being built. See [CHANGELOG.md](CHANGELOG.md).

Planned sources:

- **synthetic**: realistic generator (1/f background, alpha, mu/beta with controllable
  desynchronization, optional artifacts), deterministic with a seed
- **replay**: public datasets (via MOABB/MNE) and recorded files, played back in real time
- **lsl**: relay of an existing third-party LSL stream
- **brainflow**: any board supported by BrainFlow
- **ironbci**: IronBCI-8 headset over BLE (experimental)

## Installation

Requires Python 3.11+.

```bash
pip install openmind-plug                      # core
pip install "openmind-plug[replay]"            # + public dataset replay (MOABB)
pip install "openmind-plug[brainflow,ironbci]" # + hardware drivers
```

From source, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/openmind-bci/plug.git
cd plug
uv sync
uv run openmind-plug --help
```

## Quick start without a headset

Stream the synthetic source, controlled from the keyboard
(`a` left, `d` right, `s` feet, `r` rest, `e` eyes, `b` blink, `q` quit):

```bash
openmind-plug stream --source synthetic --montage A
```

Or replay a public dataset in real time (needs the `replay` extra):

```bash
openmind-plug stream --source replay --dataset bnci2014_001 --subject 1
```

In another terminal, check the stream:

```bash
openmind-plug check-stream
```

Any LSL consumer (OpenMind core, [LabRecorder](https://github.com/labstreaminglayer/App-LabRecorder),
MNE-LSL...) can then pick up the `openmind-eeg` stream, described in the
[stream specification](docs/stream-spec.md).

| Command | Purpose |
|---|---|
| `openmind-plug list-sources` | List sources and whether they can be used |
| `openmind-plug stream --source …` | Publish a source as the standard LSL streams |
| `openmind-plug check-stream` | Check a live stream against the specification |

Run any command with `--help` for its options.

## Data and privacy

EEG recordings are health data. Plug processes everything **locally**, never sends
data to a remote service, and stores downloads and recordings outside the
repository, in the directory given by `OPENMIND_DATA_DIR` (default: `~/openmind-data`).
Subjects are identified only by pseudonyms (`P01`, `P02`...).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE).
