"""Single-key terminal input, without pressing Enter (Unix and Windows).

Used by ``openmind-plug stream`` to drive the synthetic source from the keyboard.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager


def stdin_is_terminal() -> bool:
    """Whether standard input is an interactive terminal."""
    return sys.stdin is not None and sys.stdin.isatty()


@contextmanager
def key_reader() -> Iterator[Callable[[float], str | None]]:
    """Put the terminal in single-key mode and yield a ``read(timeout)`` function.

    ``read(timeout)`` returns the key pressed within ``timeout`` seconds, or ``None``.
    Ctrl+C keeps working. The terminal is restored on exit.
    """
    if sys.platform == "win32":
        import msvcrt
        import time

        def read_windows(timeout: float) -> str | None:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if msvcrt.kbhit():
                    return msvcrt.getwch()
                time.sleep(0.01)
            return None

        yield read_windows
    else:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)  # keys arrive one by one; Ctrl+C still raises SIGINT

        def read_unix(timeout: float) -> str | None:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            return sys.stdin.read(1) if ready else None

        try:
            yield read_unix
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def run_key_loop(
    read: Callable[[float], str | None],
    on_key: Callable[[str], bool],
    stop: threading.Event,
    poll: float = 0.1,
) -> None:
    """Call ``on_key`` for every key until ``stop`` is set or ``on_key`` returns ``False``."""
    while not stop.is_set():
        key = read(poll)
        if key is not None and not on_key(key):
            stop.set()
