"""OpenMind Plug: read any EEG source and stream it as standard LSL."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("openmind-plug")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]
