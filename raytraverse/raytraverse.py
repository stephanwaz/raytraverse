# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""Main module."""
import shlex
from subprocess import Popen, PIPE
import numpy as np
import pywt
from raytraverse import optic, translate, io


def render(scn, uv, pos, sres, rtopts='-n 12 -ab -1 -ap DFpm2.gpm 1000',
           cal='scbins.cal', bn=64):
    xyz = translate.uv2xyz(uv, axes=(0, 2, 1))
    vecs = np.hstack((pos, xyz))
    rc = (f"rcontrib_pm -V+ -fff {rtopts} -h  -e 'side:{sres[4]}' "
          f"-f {cal} -b bin -bn {bn} -m skyglow {scn}")
    # print(rc)
    p = Popen(shlex.split(rc), stdout=PIPE,
              stdin=PIPE).communicate(io.np2bytes(vecs))
    lum = optic.rgb2lum(io.bytes2np(p[0], (-1, 3)))
    return lum


def get_uniform_sampling_rate(x, t0, t1):
    """probability of random resample
    (temperature cools over time x in (0-1)"""
    return (t0 - t1)*(x - 1)**2 + t1


def get_sample_count(x, ssize, min_srate):
    """number of new samples to draw
    (temperature cools over time x in (0-1)"""
    # non parametric version:
    # Tle = (l + .01)*level_error
    # nsampc = int(np.sum(nsamps > np.min(np.std(samps, axis=(1, 2))*Tle)))
    return int(np.exp(x*np.log(min_srate))*ssize)


def get_detail(samps, axes):
    """run high pass filter over given axes"""

    # filterbank with pad and offset slice centers distribution around variance
    wav = pywt.Wavelet('custom', ([.5, 1, .5, 0], [-.5, 1, -.5, 0],
                                  [0, .5, 1, .5], [-.5, 1, .5, 0]))
    padding = [(2, 2) if i in axes else (0,0) for i in range(len(samps.shape))]
    s13 = slice(1, -3, None)
    snn = slice(None, None, None)
    slicing = tuple([s13 if i - 1 in axes else snn
                     for i in range(len(samps.shape) + 1)])
    psamps = pywt.pad(samps, padding, mode='symmetric')

    # calculate horiz, vert and diagonal detail
    d = pywt.swtn(psamps, wav, 1, trim_approx=True, axes=axes)
    d = np.asarray(tuple(d[1].values()))[slicing]

    # sum over detail and normalize (useful for non parametric sampling rates)
    # the detail can be read as delta luminance around that pixel
    d_det = np.sum(np.abs(d), 0).flatten()*(1/len(axes))
    return np.where(np.isfinite(d_det), d_det, np.nanmean(d_det))


def draw_sensor_pdf(samps, pres, dres, t0=.1, t1=.001,
                    ssize_coef=0.0, min_srate=.05):

    # detail is calculated across position and direction seperately and
    # combined by product (would be summed otherwise) to avoid drowning out
    # the signal in the more precise dimensions (assuming a mismatch in step
    # size and final stopping criteria
    p = np.ones(samps.size)
    # direction detail
    if dres[0] > samps.shape[-1]:
        daxes = tuple(range(len(pres), len(pres) + len(dres)))
        p = p * get_detail(samps, daxes)
    # position detail
    if pres[0] > samps.shape[0]:
        paxes = tuple(range(len(pres)))
        p = p*get_detail(samps, paxes)

    t = get_uniform_sampling_rate(ssize_coef, t0, t1)
    p = p*(1 - t) + np.median(p)*t
    # draw on pdf
    nsampc = get_sample_count(ssize_coef, samps.size, min_srate)
    pdraws = np.random.choice(p.size, nsampc, replace=False, p=p/np.sum(p))
    return pdraws, p
