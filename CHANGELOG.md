# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Source interface: `ChannelInfo`, `StreamInfo` (validated, immutable) and the
  `EEGSource` base class with context-manager support and output checks.
- LSL relay source (`--source lsl`): republish a third-party LSL stream in
  standard form, with label renaming and normalization, unit detection or
  override, and montage selection.
- Signal quality and `openmind-plug quality`: per-channel flat, saturated,
  line-noise, noisy and weak detection on high-passed data, impedance reporting
  hook, and a guided eyes open / closed alpha test. See `docs/quality.md`.
- Recorder and `openmind-plug record`: EEG and markers saved to FIF with
  pseudonymous subject ids only, 10-10 positions, stream metadata and gap
  annotations, in `$OPENMIND_DATA_DIR/recordings/` by default. Recordings replay
  identically, markers included. See `docs/recording.md`.
- `openmind-plug stream` (synthetic with keyboard control, replay of datasets or
  files) and `openmind-plug list-sources`.
- LSL stream specification 1.0.0 (`docs/stream-spec.md`): `openmind-eeg`
  float32 stream in microvolts with XDF-style metadata (channels with 10-10
  locations, acquisition, reference, montage, `spec_version`) and the paired
  `openmind-markers` JSON stream.
- `EEGOutlet` and `stream()` to publish any source, `EEGInlet` to consume the
  standard stream, and `openmind-plug check-stream` to validate a live stream.
- Replay source: real-time playback (optionally accelerated or looped) of public
  datasets `bnci2014_001` and `physionetmi` via MOABB, or of any file MNE reads;
  montage-based channel selection with explicit errors, microvolt conversion,
  resampling, and dataset events as JSON cue markers. Downloads go to
  `$OPENMIND_DATA_DIR/mne_data`.
- Synthetic source: 1/f background, posterior alpha (eyes open/closed), mu and
  beta over C3/Cz/C4 with controllable desynchronization (`rest`, `left`,
  `right`, `feet`), optional blink, muscle and power-line artifacts, ground-truth
  markers, keyboard bindings; deterministic with a seed.
- `Marker` type and `EEGSource.read_markers()` for sources that produce events.
- Montages: 10-10 electrode positions and named montages `A` (motor imagery,
  default) and `B` (SSVEP variant). See `docs/sources.md`.
- Package skeleton: `src/` layout, `openmind-plug` command, optional extras
  `replay`, `brainflow` and `ironbci`.
- Tooling: ruff, strict mypy, pytest, pre-commit, GitHub Actions CI
  (Linux, macOS, Windows × Python 3.11, 3.12).
