# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""functions for reading and writing SortedArRays"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from raytraverse import translate
from concurrent.futures import ThreadPoolExecutor


def np2bytes(ar, dtype='<f'):
    """format ar as bytestring

    Parameters
    ----------
    ar: np.array
    dtype: str
        argument to pass to np.dtype()

    Returns
    -------
    bytes
    """
    dt = np.dtype(dtype)
    return ar.astype(dt).tobytes()


def np2bytefile(ar, f, dtype='<f', close=False):
    """write binary data to output f

    Parameters
    ----------
    ar: np.array
    f: IOBase
        file object to write array to
    dtype: str
        argument to pass to np.dtype()
    close: bool
        whether to close f before returning

    Returns
    -------
    ar.shape
        necessary for reconstruction
    """
    f.write(np2bytes(ar, dtype))
    if close:
        f.close()
    return ar.shape


def bytes2np(buf, shape, dtype='<f'):
    """format ar as bytestring

    Parameters
    ----------
    buf: bytes, str
    shape: tuple
        array shape
    dtype: str
        argument to pass to np.dtype()

    Returns
    -------
    np.array
    """
    dt = np.dtype(dtype)
    return np.frombuffer(buf, dtype=dt).reshape(*shape)


def bytefile2np(f, shape, dtype='<f'):
    """write binary data to output f

    Parameters
    ----------
    f: IOBase
        file object to read array from
    shape: tuple
        array shape
    dtype: str
        argument to pass to np.dtype()

    Returns
    -------
    ar.shape
        necessary for reconstruction
    """
    return bytes2np(f.read(), shape, dtype)


def _write_npy(pi, uv, vals, sres, prefix):
    output = np.vstack((pi.T, uv.T, vals.T)).T
    sb = np.arange(sres[4]**2)
    output = np.vstack((np.concatenate(((0, 0, 0, 0), sb)).reshape(1, -1), output))
    shapetxt0 = "_".join([f'{i:04d}' for i in sres])
    np.save(f"{prefix}_{shapetxt0}", output)


def write_npy(pi, uv, vals, sres, prefix="UVdata", wait=False):
    if wait:
        _write_npy(pi, uv, vals, sres, prefix)
    else:
        executor = ThreadPoolExecutor()
        executor.submit(_write_npy, pi, uv, vals, sres, prefix)


def imshow(ax, im, **kwargs):
    ax.imshow(np.flip(im, 1).T, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])


def mk_img_setup(lums, decades=7, maxl=-1):
    lums = np.where(np.isfinite(lums), lums, -decades + maxl)
    fig, ax = plt.subplots(1, 1, figsize=[20, 10])
    norm = Normalize(vmin=-decades + maxl, vmax=maxl)
    lev = np.arange(-decades, 0, decades/200)
    return lums, fig, ax, norm, lev


def mk_img_uv(lums, uv, decades=7, maxl=-1, colors='viridis', mark=True):
    lums, fig, ax, norm, lev = mk_img_setup(lums, decades=decades, maxl=maxl)
    ax.tricontourf(uv[:, 0], uv[:, 1], lums, cmap=colors, norm=norm,
                   levels=lev, extend='both')
    if mark:
        ax.plot(uv[:, 0], uv[:, 1], 'or')
    return fig, ax


def mk_img_fish(lums, uv, decades=7, maxl=-1, colors='viridis', mark=True):
    lums, fig, ax, norm, lev = mk_img_setup(lums, decades=decades, maxl=maxl)
    d = uv[:, 0] > 1
    dn = np.logical_not(d)
    uv = translate.uv2xy(uv)
    uv[d, 0] += 2
    ax.tricontourf(uv[d,0], uv[d,1], lums[d], cmap=colors, norm=norm,
                   levels=lev, extend='both')
    ax.tricontourf(uv[dn, 0], uv[dn, 1], lums[dn], cmap=colors, norm=norm,
                   levels=lev, extend='both')
    if mark:
        ax.plot(uv[:, 0], uv[:, 1], 'or')
    return fig, ax
