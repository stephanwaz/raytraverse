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
from matplotlib import rcParams
from matplotlib.colors import Normalize


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
    """read binary data from f

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


def write_npy(ptidx, vecs, vals, outf):
    """write vectors and values to numpy binary file

    Parameters
    ----------
    ptidx: np.array
        shape (N,) point indices
    vecs: np.array
        shape (N, 6), x,y,z,dx,dy,dz
    vals: np.array
        shape (N, S) where S is the number of sky bins
    outf: str
        basename of file to write

    Returns
    -------
    None
    """
    output = np.vstack((ptidx.reshape(1, -1), vecs.T, vals.T)).T
    np.save(outf, output)


def imshow(ax, im, **kwargs):
    ax.imshow(im.T, origin='lower', **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])


def mk_img_setup(lums, decades=7, maxl=-1, figsize=[20, 10], ext=1):
    defparam = {
        'font.weight': 'ultralight',
        'font.family': 'sans-serif',
        'font.size': 15,
        'axes.linewidth': 0,
        }
    rcParams.update(defparam)
    lums = np.where(np.isfinite(lums), lums, -decades + maxl)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    norm = Normalize(vmin=-decades + maxl, vmax=maxl)
    lev = np.linspace(-decades + maxl, maxl, 200)
    ax.set(xlim=(-ext, ext), ylim=(-ext, ext))
    return lums, fig, ax, norm, lev


def mk_img_uv(lums, uv, decades=7, maxl=-1, colors='viridis', mark=True):
    lums, fig, ax, norm, lev = mk_img_setup(lums, decades=decades, maxl=maxl)
    ax.tricontourf(uv[:, 0], uv[:, 1], lums, cmap=colors, norm=norm,
                   levels=lev, extend='both')
    if mark:
        ax.plot(uv[:, 0], uv[:, 1], 'or')
    return fig, ax


def mk_img(lums, uv, decades=7, maxl=-1, colors='viridis', mark=True,
           figsize=[10, 10], inclmarks=None, ext=1, title=None, outf=None):
    lums, fig, ax, norm, lev = mk_img_setup(lums, decades=decades, maxl=maxl,
                                            figsize=figsize, ext=ext)
    ax.tick_params(length=10, width=.5, direction='inout', pad=5)
    ticks = np.linspace(-ext, ext, 7)
    labs = np.round(np.linspace(-ext*180/np.pi,
                                            ext*180/np.pi, 7)).astype(int)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs)
    ax.tricontourf(uv[:, 0], uv[:, 1], lums, cmap=colors, norm=norm,
                   levels=lev, extend='both')
    if mark:
        ax.scatter(uv[:inclmarks, 0], uv[:inclmarks, 1], s=10, marker='o',
                   facecolors='none', edgecolors='w', linewidths=.5)
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    if outf is None:
        plt.show()
    else:
        plt.savefig(outf)
    plt.close(fig)
    return outf
