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
from scipy.ndimage import correlate


def get_detail(samps, f1=None, f2=None, f3=None):
    d_det = []
    if f1 is None:
        # prewitt operator
        f1 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])/3
        f2 = f1.T
    s = samps.reshape(-1, *samps.shape[-2:])
    for d in s:
        ds = np.abs(correlate(d, f1))
        for f in (f2, f3):
            if f is not None:
                ds += np.abs(correlate(d, f))
        if f1.shape[0] == 2:
            ds[:-1, :-1] += ds[1:, 1:]
        d_det.append(ds.ravel())
    return np.concatenate(d_det)


def from_pdf(pdf, threshold, lb=.5, ub=4):
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
