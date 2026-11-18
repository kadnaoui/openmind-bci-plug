# Contributing to OpenMind Plug

Thanks for your interest! Bug reports, new headset drivers and documentation
improvements are all welcome.

## Development setup

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/openmind-bci/plug.git
cd plug
uv sync                       # creates .venv with runtime + dev dependencies
uv run pre-commit install     # run the checks automatically before each commit
```

Optional extras for driver work:

```bash
uv sync --extra replay --extra brainflow --extra ironbci
```

## Checks

Every change must pass:

```bash
uv run ruff check             # lint
uv run ruff format --check    # formatting
uv run mypy                   # strict typing on src/
uv run pytest -m "not slow"   # fast tests
```

Tests marked `slow` download public datasets or run for a long time; run them
with `uv run pytest -m slow`.

## Guidelines

- Code, comments, docstrings and docs are written in **English**.
- Public functions and classes are typed and documented.
- New behavior comes with tests. Hardware-dependent code is tested against a
  simulated device or recorded packets.
- Timestamps come from the LSL clock (`pylsl.local_clock()`), never from `time.time()`.
- Changes to [`docs/stream-spec.md`](docs/stream-spec.md) change the public contract:
  bump its version and describe the change in `CHANGELOG.md`.
- **Never commit EEG data** (`.fif`, `.xdf`, `.edf`, recordings, personal profiles),
  model weights or secrets. Use `OPENMIND_DATA_DIR` for local data.

## Commit messages

One commit per coherent change. The first line is a short sentence in the
imperative mood, capitalized, at most 72 characters, without a final period
or prefix:

```
Add synthetic EEG source
Fix microvolt conversion in replay source
```

Add a blank line and a short body explaining *why* when useful.

## Pull requests

Open a pull request against `main`, describe the change and how you tested it,
and make sure CI is green.
