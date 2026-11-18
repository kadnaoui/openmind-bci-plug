# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Source interface: `ChannelInfo`, `StreamInfo` (validated, immutable) and the
  `EEGSource` base class with context-manager support and output checks.
- Montages: 10-10 electrode positions and named montages `A` (motor imagery,
  default) and `B` (SSVEP variant). See `docs/sources.md`.
- Package skeleton: `src/` layout, `openmind-plug` command, optional extras
  `replay`, `brainflow` and `ironbci`.
- Tooling: ruff, strict mypy, pytest, pre-commit, GitHub Actions CI
  (Linux, macOS, Windows × Python 3.11, 3.12).
