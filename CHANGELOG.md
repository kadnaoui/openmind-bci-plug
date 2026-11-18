# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Source interface: `ChannelInfo`, `StreamInfo` (validated, immutable) and the
  `EEGSource` base class with context-manager support and output checks.
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
