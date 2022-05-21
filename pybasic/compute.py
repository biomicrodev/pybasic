from pathlib import Path
from typing import Union, List, Tuple, Optional, Set

import PIL.Image
import dask
import numpy as np
import skimage.transform
import tifffile
from dask import array as da
from numpy import typing as npt
from skimage import img_as_float

from pybasic.basic import basic
from pybasic.utils import _validate_iter_dims

ShapeLike = Union[List[int], Tuple[int, ...]]


def resize(im: npt.NDArray, shape: ShapeLike) -> npt.NDArray:
    return skimage.transform.resize(im, output_shape=shape, order=1, mode="symmetric")


def read_images(
    paths: List[Path], *, working_size: int, iter_dims: Optional[Set[int]] = None
) -> da.Array:
    """
    It's possible that the error messages on a raised exception might not make sense.

    TODO: Check what kinds of error messages are raised on invalid inputs, e.g. unequal
    array shapes.
    """
    if iter_dims is None:
        iter_dims = {}

    im1 = tifffile.imread(paths[0])

    # we resize the image axes to the working size
    shape = [s if i in iter_dims else working_size for i, s in enumerate(im1.shape)]

    # load, transform, reshape, and stack using dask
    images = [dask.delayed(tifffile.imread)(path) for path in paths]
    images = [dask.delayed(img_as_float)(im) for im in images]
    images = [dask.delayed(resize)(im, shape) for im in images]
    images = [da.from_delayed(i, shape=shape, dtype=float) for i in images]
    images = da.stack(images)
    return images


def _is_image(p: Path) -> bool:
    try:
        fp = PIL.Image.open(p)
    except PIL.UnidentifiedImageError:
        is_image = False
    else:
        fp.close()
        is_image = True

    return is_image


def compute(
    paths: List[Path],
    *,
    iter_dims: Optional[List[int]] = None,
    rgb: bool = False,
    working_size: int = 128,
    flatfield_reg: Optional[float] = None,
    darkfield_reg: Optional[float] = None,
    compute_darkfield: bool = False,
) -> Tuple[npt.NDArray, npt.NDArray]:
    # check if image by trying to open it with pillow; could be slow?
    # is it performant to do this rather than have a dask worker try and fail?
    # we can check the file extension, but those can be unreliable
    paths = [p for p in paths if _is_image(p)]
    if len(paths) == 0:
        raise ValueError("No images provided/found")
    elif len(paths) == 1:
        raise ValueError("One image is not sufficient for illumination correction")

    # get first image
    orig_im_shape = tifffile.imread(paths[0]).shape

    # reconcile first image with provided arguments
    # for now, rgb is interpreted as just a special case of multichannel
    iter_dims = set(iter_dims) if iter_dims is not None else set()
    iter_dims = _validate_iter_dims(orig_im_shape, rgb=rgb, iter_dims=iter_dims)

    # read_images
    stack = read_images(paths, working_size=working_size, iter_dims=iter_dims)
    print(f"Image stack with shape {stack.shape}")

    if len(iter_dims) == 0:
        # nothing to iterate over, so we pass entire array to basic
        stack = np.asarray(stack)
        assert stack.ndim == 3
        flatfield, darkfield = basic(
            stack,
            flatfield_reg=flatfield_reg,
            darkfield_reg=darkfield_reg,
            compute_darkfield=compute_darkfield,
        )

    else:
        """
        This is a little awkward with dimension wrangling because I would like to keep
        the dimension order of the original images the same, and I need to support an
        arbitrary set of iter dims (as far as I'm concerned). We could probably make
        the code more legible just by moving dims around before basic, and move them
        back afterwards.
        """

        def func(a: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
            if a.size == 1:
                # dask (or gufunc) runs this function at least once with an array of
                # size 1, just to check what the output is. because this function is
                # computationally expensive, we provide dummy arrays with the same dtype
                return (
                    np.zeros(shape=(), dtype=a.dtype),
                    np.zeros(shape=(), dtype=a.dtype),
                )

            # remove, but keep track of, singular dimensions
            sing_dims = [i for i in range(a.ndim) if a.shape[i] == 1]
            a = a.squeeze()

            flatfield, darkfield = basic(
                a,
                flatfield_reg=flatfield_reg,
                darkfield_reg=darkfield_reg,
                compute_darkfield=compute_darkfield,
            )

            # recover original shape
            flatfield = np.expand_dims(flatfield, sing_dims)
            darkfield = np.expand_dims(darkfield, sing_dims)

            return flatfield, darkfield

        # append one to the iter dims
        # TODO: is it worth using dstack vs stack to avoid shifting between image dims
        # and stack dims?
        stack_iter_dims = [ax + 1 for ax in sorted(list(iter_dims))]

        # rechunk, because gufunc applies func one chunk at a time
        chunksize = [None if i not in stack_iter_dims else 1 for i in range(stack.ndim)]
        stack = stack.rechunk(chunksize)

        # specify input and output dims to gufunc
        # input_dims are relative to stack
        input_dims = tuple(i for i in range(stack.ndim) if i not in stack_iter_dims)
        assert len(input_dims) == 3
        # output_dims are relative to image, and since we lose the first dimension after
        # applying basic, we have to shift by one
        output_dims = tuple(
            i - 1 for i in range(1, stack.ndim) if i not in stack_iter_dims
        )
        assert len(output_dims) == 2

        res = da.apply_gufunc(
            func,
            "(c,i,j) -> (i,j), (i,j)",
            stack,
            axes=[input_dims, output_dims, output_dims],
            allow_rechunk=True,
        )
        flatfield, darkfield = dask.compute(*res)

    # resize back to original shape
    flatfield = resize(flatfield, orig_im_shape)
    if compute_darkfield:
        darkfield = resize(darkfield, orig_im_shape)

    return flatfield, darkfield
