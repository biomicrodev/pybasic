import warnings
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from bmd_perf.profiling import timed_ctx

from .linalg import fro_norm, dct2d, idct2d, l1_norm

rng = np.random.default_rng(seed=0)


def scalar_shrink(arr: npt.NDArray, epsilon: float) -> npt.NDArray:
    return np.sign(arr) * np.maximum(np.abs(arr) - epsilon, 0)


def inexact_alm_rspca_l1(
    stack: npt.NDArray,
    *,
    flatfield_reg: float,
    darkfield_reg: float,
    optim_tol: float,
    max_iters: int,
    weights: Optional[npt.NDArray] = None,
    compute_darkfield=False,
    # darkfield_upper_lim: float = 1e7,
) -> Tuple:
    # stack is of shape NYX
    assert stack.ndim == 3
    n = stack.shape[0]
    im_dims = stack.shape[1:]
    full_dims = stack.shape

    if weights is not None:
        if weights.shape != stack.shape:
            raise ValueError(
                f"Mismatch between dims of weight ({weights.shape}) and images ({stack.shape})"
            )
        weights = np.ones(stack.shape)

    lm1 = 0  # lagrange multiplier
    # lm2 = 0
    ent1 = 1  # ?
    ent2 = 10

    # variables
    im_res = np.zeros(full_dims)
    base = np.ones((n, 1, 1))
    dark_res = np.zeros(im_dims)
    flat = np.zeros(im_dims)

    # adaptive penalty
    ims_vec = stack.reshape((n, -1))
    norm_two = np.linalg.svd(ims_vec, compute_uv=False, full_matrices=False)[0]
    pen = 12.5 / norm_two  # initial penalty
    pen_max = pen * 1e7
    pen_mult = 1.5

    # convenience constants
    ims_norm = fro_norm(stack)
    ims_min = stack.min()

    dark_mean = 0
    # A_upper_lim = ims.min(axis=0)
    # A_inmask = np.zeros((h, w))
    # A_inmask[h // 6 : h // 6 * 5, w // 6 : w // 6 * 5] = 1

    # temp_sub = np.zeros(shape=(3,) + stack.shape, dtype=stack.dtype)

    it = 0
    while True:
        # update flatfield
        im_base = base * flat + dark_res

        _diff = (stack - im_base - im_res + lm1 / pen) / ent1
        _diff = _diff.mean(axis=0)

        flat_f = dct2d(flat) + dct2d(_diff)
        flat_f = scalar_shrink(flat_f, flatfield_reg / (ent1 * pen))
        flat = idct2d(flat_f)

        # update residual
        im_base = base * flat + dark_res
        im_res += (stack - im_base - im_res + lm1 / pen) / ent1
        im_res = scalar_shrink(im_res, weights / (ent1 * pen))

        # update baseline
        im_diff = stack - im_res
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
        im_diff = stack - im_base - im_res
        lm1 += pen * im_diff

        # update penalty
        pen = min(pen * pen_mult, pen_max)

        # check for stop condition
        it += 1

        is_converged = (fro_norm(im_diff) / ims_norm) < optim_tol
        if is_converged:
            break

        if not is_converged and it >= max_iters:
            warnings.warn("Maximum iterations reached")
            break

    dark_res += dark_mean * flat

    if not compute_darkfield:
        assert np.all(dark_res == 0)

    # for reporting
    iter_report = dict(iter=it, max_iters=max_iters)

    return im_base, im_res, dark_res, iter_report


def basic(
    stack: npt.NDArray,
    *,
    flatfield_reg: Optional[float] = None,
    darkfield_reg: Optional[float] = None,
    optim_tol=1e-6,
    max_iters=500,
    compute_darkfield=False,
    eps=0.1,
    reweight_tol=1e-3,
    max_reweight_iters=10,
    verbose=False,
    sort=False,
) -> Tuple[npt.NDArray, npt.NDArray]:
    # validate inputs
    assert stack.ndim == 3, "Images must be 3D (NYX)"

    # resize to working size
    full_dims = stack.shape
    im_dims = stack.shape[1:]

    # apply automatic regularization coefficient strategy
    if flatfield_reg is None or darkfield_reg is None:
        ims_norm = stack.mean(axis=0)
        ims_norm /= ims_norm.mean()
        ims_dct = dct2d(ims_norm)
        ims_l1 = l1_norm(ims_dct)

        if flatfield_reg is None:
            flatfield_reg = ims_l1 / 800
        if darkfield_reg is None:
            darkfield_reg = ims_l1 / 2_000

    if verbose:
        if compute_darkfield:
            print(
                f"Flatfield reg: {flatfield_reg:,.3f}, "
                f"darkfield reg: {darkfield_reg:,.3f}"
            )
        else:
            print(f"Flatfield reg: {flatfield_reg:,.3f}")

    if sort:
        """
        Oddly enough, although sorting is in the paper, if the images are sorted, the
        resulting flatfield image contains high frequencies; otherwise, the flatfield
        image is smooth as expected. We leave it out by default.
        """
        stack = np.sort(stack, axis=0)  # along index dimension

    weight = np.ones(full_dims)
    flat_last = np.ones(im_dims)
    dark_last = rng.standard_normal(im_dims)

    it = 0
    while True:
        _timer = timed_ctx(f"Re-weighting iter {it + 1}").__enter__()

        im_base, im_res, dark, iter_report = inexact_alm_rspca_l1(
            stack,
            flatfield_reg=flatfield_reg,
            darkfield_reg=darkfield_reg,
            optim_tol=optim_tol,
            max_iters=max_iters,
            weights=weight,
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

        _timer.__exit__()

        it += 1

        if verbose:
            print(
                f"Reweight iters: {it:,}/{max_reweight_iters:,}, "
                f"optim iters: {iter_report['iter']:,}/{iter_report['max_iters']:,}; "
                f"{_timer._elapsed.to_compact():~,.1f}"
            )

        # check stop conditions
        is_converged = max(flat_mad, dark_mad) <= reweight_tol
        if is_converged:
            if verbose:
                print("Converged")
            break

        if it >= max_reweight_iters:
            warnings.warn("Reached max iterations; stopping")
            break

        # update persistent variables
        flat_last = flat_curr
        dark_last = dark_curr

    flat = im_base.mean(axis=0) - dark
    flat /= flat.mean()
    return flat, dark


__all__ = ["basic"]
