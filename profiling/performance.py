from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt
import skimage.transform
import skimage.util
from tifffile import tifffile

from pybasic.basic import basic
from pybasic.utils import timed_ctx

working_size = 128
images_path = Path(r"../examples/images/Cell_culture/Uncorrected/DAPI")


def resize(a: npt.NDArray, shape: Tuple[int, ...]) -> npt.NDArray:
    return skimage.transform.resize(a, shape, order=1, mode="symmetric")


with timed_ctx("read in images", print):
    images = [tifffile.imread(p) for p in images_path.iterdir()]
    images = [resize(im, (working_size, working_size)) for im in images]
    images = np.stack(images)
    images = skimage.util.img_as_float(images)

with timed_ctx("run", print):
    flatfield, darkfield = basic(images, compute_darkfield=False)
