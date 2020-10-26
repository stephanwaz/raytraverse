# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""wavelet and associated probability functions."""
import numpy as np
import pywt
from raytraverse.craytraverse import from_pdf as c_from_pdf


def get_detail(samps, axes):
    """run high pass filter over given axes"""

    # filterbank with pad and offset slice centers distribution around variance
    wav = pywt.Wavelet('custom4', ([0.40824829, 0.81649658, 0.40824829, 0.],
                                   [-0.40824829, 0.81649658, -0.40824829, 0],
                                   [0, 0.40824829, 0.81649658, 0.40824829],
                                   [-0.40824829, 0.81649658, 0.40824829, 0]))
    # mod adds extra padding to ensure evenness of transformed dimensions
    padding = [(2, 2 + int(np.mod(s, 2))) if i in axes else (0, 0) for i, s in
               enumerate(samps.shape)]
    snn = slice(None, None, None)
    slicing = (snn, ) + tuple([slice(1, -3 - int(np.mod(s, 2)), None)
                               if i in axes else snn for i, s in
                               enumerate(samps.shape)])
    psamps = pywt.pad(samps, padding, mode='symmetric')
    # calculate horiz, vert and diagonal detail
    d = pywt.swtn(psamps, wav, 1, trim_approx=True, axes=axes)
    d = np.asarray(tuple(d[1].values()))[slicing]
    # sum over detail and normalize (useful for non parametric sampling rates)
    # the detail can be read as delta luminance around that pixel
    d_det = np.sum(np.abs(d), 0).ravel() * 3
    m = np.nanmean(d_det)
    return np.where(np.isfinite(d_det), d_det, m)


def from_pdf(pdf, threshold):
    candidates, bidx, nsampc = c_from_pdf(pdf, threshold)
    if nsampc == 0:
        return bidx
    # if normalization happens in c-func floating point precision does not
    # guarantee that pdfc adds to 1, which choice() requires.
    pdfc = pdf[candidates]/np.sum(pdf[candidates])
    cidx = np.random.default_rng().choice(candidates, nsampc,
                                          replace=False, p=pdfc)
    return np.concatenate((bidx, cidx))
