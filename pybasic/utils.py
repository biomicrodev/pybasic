import time

from typing import Callable


class timed_ctx:
    def __init__(self, msg: str, out: Callable[[str], None]):
        self._msg = msg
        self._out = out

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        t1 = time.perf_counter()
        self._out(f"{self._msg}: {(t1 - self._t0) * 1000:,.1f}ms")
