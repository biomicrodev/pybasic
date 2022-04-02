from pathlib import Path
from typing import List, Set, Optional, Tuple, Union

import dask
import dask.array as da
import numpy.typing as npt
import skimage.transform
import tifffile
from skimage.util import img_as_float

ShapeLike = Union[List[int], Tuple[int, ...]]


def resize(im: npt.NDArray, shape: ShapeLike) -> npt.NDArray:
    return skimage.transform.resize(im, output_shape=shape, order=1, mode="symmetric")


def read_images(
    paths: List[Path], *, working_size: int, iter_dims: Optional[Set[int]] = None
) -> da.Array:
    """
    It's possible that the error messages on a raised exception might not make sense.
    Check what kinds of error messages are raised on invalid inputs, e.g. unequal array
    shapes.
    """
    if iter_dims is None:
        iter_dims = {}

    path1 = paths[0]
    im1 = tifffile.imread(path1)

    # we resize the image axes to the working size
    shape = [s if i in iter_dims else working_size for i, s in enumerate(im1.shape)]

    images = [dask.delayed(tifffile.imread)(path) for path in paths]
    images = [dask.delayed(img_as_float)(im) for im in images]
    images = [
        dask.delayed(lambda im: resize(im, shape), name=resize.__name__)(im)
        for im in images
    ]
    images = [da.from_delayed(i, shape=shape, dtype=float) for i in images]
    images = da.stack(images)
    return images
