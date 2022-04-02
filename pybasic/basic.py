import warnings
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from pybasic.linalg import dct2d, idct2d, fro_norm, l1_norm
from pybasic.utils import timed_ctx

rng = np.random.default_rng(seed=0)


def scalar_shrink(arr: npt.NDArray, epsilon: float) -> npt.NDArray:
    return np.sign(arr) * np.maximum(np.abs(arr) - epsilon, 0)


def inexact_alm_rspca_l1(
    ims: npt.NDArray,
    *,
    flatfield_reg: float,
    darkfield_reg: float,
    optim_tol: float,
    max_iters: int,
    weight: Optional[npt.NDArray] = None,
    compute_darkfield: bool = False,
    # darkfield_upper_lim: float = 1e7,
) -> Tuple[npt.NDArray, ...]:
    if weight is not None:
        if weight.shape != ims.shape:
            raise ValueError(
                f"Mismatch between dims of weight ({weight.shape}) and images ({ims.shape})"
            )
        weight = np.ones(ims.shape)

    lm1 = 0  # lagrange multiplier
    # lm2 = 0
    ent1 = 1  # ?
    ent2 = 10

    # convenience shapes
    n = ims.shape[0]
    im_dims = ims.shape[1:3]
    full_dims = ims.shape

    # variables
    im_res = np.zeros(full_dims)
    base = np.ones((n, 1, 1))
    dark_res = np.zeros(im_dims)
    flat = np.zeros(im_dims)

    # adaptive penalty
    ims_vec = ims.reshape((n, -1))
    norm_two = np.linalg.svd(ims_vec, compute_uv=False, full_matrices=False)[0]
    pen = 12.5 / norm_two  # initial penalty
    pen_max = pen * 1e7
    pen_mult = 1.5
    # TODO: can we set the initial penalty to something reasonable without having to
    # compute the SVD?

    # convenience constants
    ims_norm = fro_norm(ims)
    ims_min = ims.min()

    dark_mean = 0
    # A_upper_lim = ims.min(axis=0)
    # A_inmask = np.zeros((h, w))
    # A_inmask[h // 6 : h // 6 * 5, w // 6 : w // 6 * 5] = 1

    it = 0
    while True:
        # update flatfield
        im_base = base * flat + dark_res
        _diff = (ims - im_base - im_res + lm1 / pen) / ent1
        _diff = _diff.mean(axis=0)
        flat_f = dct2d(flat) + dct2d(_diff)
        flat_f = scalar_shrink(flat_f, flatfield_reg / (ent1 * pen))
        flat = idct2d(flat_f)

        # update residual
        im_base = base * flat + dark_res
        im_res += (ims - im_base - im_res + lm1 / pen) / ent1
        im_res = scalar_shrink(im_res, weight / (ent1 * pen))

        # update baseline
        im_diff = ims - im_res
        base = im_diff.mean(axis=(1, 2), keepdims=True) / im_diff.mean()
        base[base < 0] = 0

        if compute_darkfield:
            # calculate dark_mean using least-square fitting
            base_valid_ind = (base < 1).squeeze()
            im_diff_valid = im_diff[base_valid_ind]
            base_valid = base[base_valid_ind]

            flat_mean = flat.mean()
            _flat_gt = flat > (flat_mean - 1e-6)
            _flat_lt = flat < (flat_mean + 1e-6)
            diff = (im_diff_valid[:, _flat_gt].mean(axis=-1)) - (
                im_diff_valid[:, _flat_lt].mean(axis=-1)
            )

            n_valid = np.count_nonzero(base_valid_ind)
            _temp1 = np.square(base_valid).sum()
            _temp2 = base_valid.sum()
            _temp3 = diff.sum()
            _temp4 = (base_valid * diff).sum()
            _temp5 = _temp2 * _temp3 - n_valid * _temp4
            if _temp5 == 0:
                dark_mean = 0
            else:
                dark_mean = (_temp1 * _temp3 - _temp2 * _temp4) / _temp5

            # clip to reasonable values
            dark_mean = max(dark_mean, 0)
            dark_mean = min(dark_mean, ims_min / flat_mean)

            # optimize dark_res
            _diff = (im_diff_valid - base_valid * flat).mean(axis=0)
            dark_diff = dark_mean * (flat_mean - flat)
            dark_res = _diff - _diff.mean() - dark_diff

            dark_res_f = dct2d(dark_res)
            dark_res_f = scalar_shrink(dark_res_f, darkfield_reg / (ent2 * pen))
            dark_res = idct2d(dark_res_f)
            dark_res = scalar_shrink(dark_res, darkfield_reg / (ent2 * pen))

            dark_res += dark_diff

        # update lagrangian multiplier
        # if I'm understanding Table 1 in the supplementary section correctly, the next
        # line is missing from the MATLAB implementation
        im_base = base * flat + dark_res
        im_diff = ims - im_base - im_res
        lm1 += pen * im_diff

        # update penalty
        pen = min(pen * pen_mult, pen_max)

        # check for stop condition
        it += 1

        is_converged = (fro_norm(im_diff) / ims_norm) < optim_tol
        if is_converged:
            print(f"Converged in {it} iterations")
            break

        if not is_converged and it >= max_iters:
            warnings.warn("Maximum iterations reached")
            break

    dark_res += dark_mean * flat

    return im_base, im_res, dark_res


def basic(
    images: npt.NDArray,
    *,
    flatfield_reg: Optional[float] = None,
    darkfield_reg: Optional[float] = None,
    optim_tol: float = 1e-6,
    max_iters: int = 500,
    compute_darkfield: bool = False,
    eps: float = 0.1,
    reweight_tol: float = 1e-3,
    max_reweight_iters: int = 10,
) -> Tuple[npt.NDArray, npt.NDArray]:
    if images.ndim != 3:
        raise ValueError("Images must be 3D (IYX)")

    ims = images
    orig_dims = ims.shape[1:3]

    # resize to working size
    full_dims = ims.shape
    im_dims = ims.shape[1:3]

    # apply automatic regularization coefficient strategy
    if flatfield_reg is None or darkfield_reg is None:
        ims_norm = ims.mean(axis=0)
        ims_norm /= ims_norm.mean()
        ims_dct = dct2d(ims_norm)
        ims_l1 = l1_norm(ims_dct)

        if flatfield_reg is None:
            flatfield_reg = ims_l1 / 800
        if darkfield_reg is None:
            darkfield_reg = ims_l1 / 2_000

    ims = np.sort(ims, axis=0)  # along index dimension

    weight = np.ones(full_dims)
    flat_last = np.ones(im_dims)
    dark_last = rng.standard_normal(im_dims)

    it = 0
    while True:
        # I don't want another indentation level, consider refactoring for this use case
        _timer = timed_ctx(f"Re-weighting iter {it+1}", print).__enter__()

        im_base, im_res, dark = inexact_alm_rspca_l1(
            ims,
            flatfield_reg=flatfield_reg,
            darkfield_reg=darkfield_reg,
            optim_tol=optim_tol,
            max_iters=max_iters,
            weight=weight,
            compute_darkfield=compute_darkfield,
        )

        # update weight
        # in the MATLAB implementation, im_base is averaged over the pixel dimension
        # when updating the weight, but this isn't explicitly mentioned in the
        # supplementary section. we'll keep the MATLAB version for now.
        _ratio = im_res / im_base.mean(axis=(1, 2), keepdims=True)
        weight = 1 / (np.abs(_ratio) + eps)
        weight *= weight.size / weight.sum()

        flat_raw = im_base.mean(axis=0) - dark
        flat_curr = flat_raw / flat_raw.mean()
        dark_curr = dark

        # check stop condition
        flat_mad = l1_norm(flat_curr - flat_last) / l1_norm(flat_last)
        temp_diff = l1_norm(dark_curr - dark_last)
        if temp_diff < 1e-7:
            dark_mad = 0
        else:
            dark_mad = temp_diff / max(l1_norm(dark_last), 1e-6)

        # stop timing before stop condition check
        _timer.__exit__()

        is_converged = max(flat_mad, dark_mad) <= reweight_tol
        if is_converged:
            print(f"Converged in {it} iterations")
            break

        it += 1
        if it >= max_reweight_iters:
            warnings.warn("Reached max iterations; stopping")
            break

        # update persistent variables
        flat_last = flat_curr
        dark_last = dark_curr

    flat = im_base.mean(axis=0) - dark
    flat /= flat.mean()

    if not compute_darkfield:
        dark = np.zeros(orig_dims)

    return flat, dark


__all__ = ["basic"]
