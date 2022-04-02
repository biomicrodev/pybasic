"""
It's a bit awkward having util functions for the scripts here. But I would really like
to test these functions, so here they are.
"""

import warnings
from pathlib import Path
from typing import Set, Iterator, Tuple


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
