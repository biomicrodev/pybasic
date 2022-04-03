import numpy as np
import numpy.typing as npt
import scipy.fft


def _is_np_2d(a: npt.NDArray):
    if a.ndim != 2:
        raise ValueError("Array should be two-dimensional!")


def dct2d(a: npt.NDArray) -> npt.NDArray:
    # We have wrappers around scipy's dct functions to ensure that norm is 'ortho'
    # for true invertibility
    _is_np_2d(a)
    return scipy.fft.dctn(a, norm="ortho")


def idct2d(a: npt.NDArray) -> npt.NDArray:
    _is_np_2d(a)
    return scipy.fft.idctn(a, norm="ortho")


def l1_norm(a: npt.NDArray) -> float:
    return np.abs(a).sum().item()


def fro_norm(a: npt.NDArray) -> float:
    # See "profiling/fro_norm.ipynb" for why this was chosen
    a = a.ravel()
    return np.sqrt(np.dot(a, a)).item()


__all__ = ["dct2d", "idct2d", "l1_norm", "fro_norm"]
