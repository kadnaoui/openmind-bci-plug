"""Local data locations.

Downloads, caches and recordings live outside the repositories, in the directory
given by the ``OPENMIND_DATA_DIR`` environment variable (default ``~/openmind-data``).
"""

from __future__ import annotations

import os
from pathlib import Path

#: Environment variable that overrides the data directory.
DATA_DIR_ENV = "OPENMIND_DATA_DIR"


def data_dir(create: bool = False) -> Path:
    """Return the OpenMind data directory.

    Args:
        create: Create the directory if it does not exist.
    """
    value = os.environ.get(DATA_DIR_ENV, "").strip()
    path = Path(value).expanduser() if value else Path.home() / "openmind-data"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
