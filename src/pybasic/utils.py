"""
It's a bit awkward having util functions for the scripts here. But I would really like
to test these functions, so here they are.
"""
import cProfile
import functools
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Iterator, Tuple, Set, Optional

from pint import UnitRegistry
from viztracer import VizTracer

ureg = UnitRegistry()


class timed_ctx:
    def __init__(self, msg: str, out: Optional[Callable[[str], None]] = None):
        self._msg = msg
        self._out = out

    def __enter__(self):
        self._t0 = time.perf_counter() * ureg.second
        return self

    def __exit__(self, *args):
        t1 = time.perf_counter() * ureg.second
        elapsed = t1 - self._t0
        if self._out is not None:
            self._out(f"{self._msg}: {elapsed.to_compact():,.2f}")
        self._elapsed = elapsed


def _flatten_paths(paths: Iterator[Path], depth=0) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            if depth >= 1:
                for p in _flatten_paths(path.iterdir(), depth - 1):
                    yield p
            # else:
            #     for p in path.iterdir():
            #         if not p.is_dir():
            #             yield p
        else:
            yield path


def _validate_iter_axes(
    shape: Tuple[int, ...], *, rgb: bool, iter_axes: Set[int]
) -> Set[int]:
    ndim = len(shape)

    if ndim == 2:
        if rgb:
            warnings.warn("Can't interpret 2D images as RGB; ignoring RGB flag")

        if len(iter_axes) >= 1:
            warnings.warn(
                f"Can't interpret iter dims ({iter_axes}) for 2D images; ignoring iter dims"
            )

        return set()

    elif ndim >= 3:
        if rgb:
            try:
                iter_axes = {shape.index(3)}
            except ValueError:
                raise ValueError(f"Unable to interpret image with shape {shape} as RGB")

        if ndim - 2 != len(iter_axes):
            raise ValueError(f"Must provide {ndim - 2} iter dims!")

        if not all(ax < ndim for ax in iter_axes):
            raise ValueError(
                f"Provided iter axes ({iter_axes}) exceed image ndims ({ndim})"
            )

        return iter_axes

    else:
        raise ValueError(f"Cannot process images of shape {shape}")


def _query_yes_no(question: str, default: Optional[str] = "no") -> bool:
    # Many thanks to this stackoverflow answer: https://stackoverflow.com/a/3041990

    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError(f"Invalid default answer: '{default}'")

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def viztrace(**viztracer_kwargs) -> Callable:
    def outer(func: Callable) -> Callable:
        @functools.wraps(func)
        def inner(*args, **kwargs):
            try:
                filename = Path(sys.modules["__main__"].__file__).stem
            except AttributeError:
                filename = "unknown"

            log_path = (
                Path(".")
                / "logs"
                / "viztracer"
                / filename
                / f"{func.__name__}_{int(time.time())}.json"
            )

            log_path.parent.mkdir(exist_ok=True, parents=True)

            tracer = VizTracer(output_file=str(log_path), **viztracer_kwargs)
            tracer.start()
            try:
                ret_val = func(*args, **kwargs)
            finally:
                tracer.stop()
                tracer.save()
                tracer.terminate()

            return ret_val

        return inner

    return outer


def profile() -> Callable:
    def outer(func: Callable) -> Callable:
        @functools.wraps(func)
        def inner(*args, **kwargs):
            try:
                filename = Path(sys.modules["__main__"].__file__).stem
            except AttributeError:
                filename = "unknown"

            log_path = (
                Path(".")
                / "logs"
                / "cprofile"
                / filename
                / f"{func.__name__}_{int(time.time())}.prof"
            )

            log_path.parent.mkdir(exist_ok=True, parents=True)

            pr = cProfile.Profile()
            pr.enable()

            try:
                ret_val = func(*args, **kwargs)

            finally:
                pr.disable()
                pr.dump_stats(log_path)
                print(f"cProfile profile saved to {log_path.resolve()}")

            return ret_val

        return inner

    return outer
