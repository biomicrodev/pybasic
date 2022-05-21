"""
It's a bit awkward having util functions for the scripts here. But I would really like
to test these functions, so here they are.
"""

import sys
import time
import warnings
from pathlib import Path

from typing import Callable, Iterator, Tuple, Set, Optional


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


def _flatten_paths(paths: Iterator[Path], depth: int = 0) -> Iterator[Path]:
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


def _validate_iter_dims(
    shape: Tuple[int, ...], *, rgb: bool, iter_dims: Set[int]
) -> Set[int]:
    ndim = len(shape)

    if ndim == 2:
        if rgb:
            warnings.warn("Can't interpret 2D images as RGB; ignoring RGB flag")

        if len(iter_dims) >= 1:
            warnings.warn(
                f"Can't interpret iter dims ({iter_dims}) for 2D images; ignoring iter dims"
            )

        return set()

    elif ndim >= 3:
        if rgb:
            try:
                iter_dims = {shape.index(3)}
            except ValueError:
                raise ValueError(f"Unable to interpret image with shape {shape} as RGB")

        if ndim - 2 != len(iter_dims):
            raise ValueError(f"Must provide {ndim - 2} iter dims!")

        if not all(ax < ndim for ax in iter_dims):
            raise ValueError(
                f"Provided iter dims ({iter_dims}) exceed image ndims ({ndim})"
            )

        return iter_dims

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
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
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
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
