# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""wavelet and associated probability functions."""
import numpy as np
from raytraverse.craytraverse import from_pdf as c_from_pdf
from scipy.ndimage import convolve


def get_detail(data, *args, mode='reflect', cval=0.0):
    """convolve a set of kernels with data. computes the sum of the
    absolute values of each convolution.

    Parameters
    ----------
    data: np.array
        source data (atleast 2D), detail calculated over last 2D
    args: np.array
        filters
    mode: str
        signal extension mode (passed to scipy.ndimage.convolve)
    cval: float
        constant value (passed to scipy.ndimage.convolve, used when
        mode='constant')

    Returns
    -------
    detail_array: np.array
        1d array of detail coefficients (row major order) matching
        size of data

    """
    d_det = []
    s = data.reshape(-1, *data.shape[-2:])
    for d in s:
        ds = np.zeros_like(d)
        for f in args:
            ds += np.abs(convolve(d, f, mode=mode, cval=cval))
        d_det.append(ds.ravel())
    return np.concatenate(d_det)


def from_pdf(pdf, threshold, lb=.5, ub=4):
    """generate choices from a numeric probability distribution

    Parameters
    ----------
    pdf: np.array
        1-d array of weights
    threshold: float
        the threshold used to determine the number of choices to draw given
        by pdf > threshold
    lb: float, optional
        values below threshold * lb will be excluded from candidates
        (lb must be in (0,1)
    ub: float, optional
        the maximum weight is set to ub*threshold, meaning all values in pdf
        >=  to ub*threshold have an equal chance of being selected.
        in cases where extreme values are much higher than moderate values,
        but 100% sampling of extreme areas should be avoided, this value should
        be lower, such as when a region is sampled at a very high resolution (
        as is the case with directional sampling). On the other hand, set this
        value higher for sampling schemes with a low final resolution (like
        area sampling). If ub <= 1, then a deterministic choice is made,
        returning the idx of all values in pdf > threshold.

    Returns
    -------
    idx: np.array
        an index array of choices, size varies.

    """
    # bypass random sampling
    if ub <= 1:
        return np.argwhere(pdf > threshold).ravel()
    pdf[pdf > ub*threshold] = ub*threshold
    candidates, bidx, nsampc = c_from_pdf(pdf, threshold, lb=lb, ub=ub+1)
    if nsampc == 0:
        return bidx
    # if normalization happens in c-func floating point precision does not
    # guarantee that pdfc adds to 1, which choice() requires.
    pdfc = pdf[candidates]/np.sum(pdf[candidates])
    cidx = np.random.default_rng().choice(candidates, nsampc,
                                          replace=False, p=pdfc)
    return np.concatenate((bidx, cidx))
