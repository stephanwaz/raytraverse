# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""functions for reading and writing SortedArRays"""
import shlex
from subprocess import Popen, PIPE
import warnings

import numpy as np
import matplotlib.pyplot as plt
from clipt.plot import get_colors
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from clipt import mplt
from matplotlib.patches import Polygon

from raytraverse import optic


def call_sampler(outf, command, vecs):
    """make subprocess call to sampler given as command, expects rgb value
    as return for each vec

    Parameters
    ----------
    outf: str
        path to write out to
    command: str
        command line with executable and options
    vecs: np.array
        vectors to pass as stdin to command

    Returns
    -------
    lums: np.array
        of length vectors.shape[0]

    """
    f = open(outf, 'a+b')
    lum_file_pos = f.tell()
    p = Popen(shlex.split(command), stdout=f, stdin=PIPE)
    p.communicate(np2bytes(vecs))
    f.seek(lum_file_pos)
    lum = optic.rgb2rad(bytes2np(f.read(), (-1, 3)))
    f.close()
    return lum


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


def imshow(im, figsize=[10, 10], outf=None, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(im.T, origin='lower', **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if outf is None:
        plt.show()
    else:
        plt.savefig(outf)
    plt.close(fig)


def mk_img_setup(lums, decades=7, maxl=-1, figsize=[20, 10], ext=1):
    defparam = {
        'font.weight': 'ultralight',
        'font.family': 'sans-serif',
        'font.size': 15,
        'axes.linewidth': 0,
        }
    rcParams.update(defparam)
    lums = np.where(np.isfinite(lums), lums, -decades + maxl)
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot()
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


def set_ang_ticks(ax, ext):
    ax.tick_params(length=10, width=.5, direction='inout', pad=5)
    ticks = np.linspace(-ext, ext, 7)
    labs = np.round(np.linspace(-ext*180/np.pi,
                                ext*180/np.pi, 7)).astype(int)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs)


def save_img(fig, ax, title, outf):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()
        if title is not None:
            ax.set_title(title)
        fig.savefig(outf)


def mk_img(lums, uv, outf, decades=7, maxl=-1, colors='viridis', mark=True,
           figsize=[10, 10], inclmarks=None, ext=1, title=None, **kwargs):
    lums, fig, ax, norm, lev = mk_img_setup(lums, decades=decades, maxl=maxl,
                                            figsize=figsize, ext=ext)
    set_ang_ticks(ax, ext)
    ax.tricontourf(uv[:, 0], uv[:, 1], lums, cmap=colors, norm=norm,
                   levels=lev, extend='both')
    if mark:
        ax.scatter(uv[:inclmarks, 0], uv[:inclmarks, 1], s=10, marker='o',
                   facecolors='none', edgecolors='w', linewidths=.5)
    save_img(fig, ax, title, outf)
    return outf


def mk_img_scatter(lums, uv, outf, decades=7, maxl=-1, colors='viridis',
           figsize=[10, 10], ext=1, title=None, **kwargs):
    lums, fig, ax, norm, lev = mk_img_setup(lums, decades=decades, maxl=maxl,
                                            figsize=figsize, ext=ext)
    set_ang_ticks(ax, ext)
    ax.scatter(uv[:, 0], uv[:, 1], marker='o', s=3, c=lums, cmap=colors, norm=norm)
    ax.set_facecolor((0,0,0))
    save_img(fig, ax, title, outf)
    return outf


def mk_img_voronoi(lums, uv, verts, regions, vi, decades=7, maxl=-1, colors='viridis', mark=True,
                   figsize=[10, 10], inclmarks=None, ext=1, title=None, outf=None, **kwargs):
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

    colormap = get_colors(colors)
    colormap.set_norm(norm)
    for v in vi:
        r = verts[regions[v]]
        polygon = Polygon(r, closed=True, lw=.5, color=colormap.to_rgba(lums[v]), zorder=-1)
        ax.add_patch(polygon)

    if mark:
        ax.scatter(uv[:inclmarks, 0], uv[:inclmarks, 1], s=10, marker='o',
                   facecolors='none', edgecolors=(1, 1, 1, .5), linewidths=.5)
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    if outf is None:
        plt.show()
    else:
        plt.savefig(outf)
    plt.close(fig)
    return outf


def hist(lums, bins='auto', outf=None, **kwargs):
    h, binedges = np.histogram(lums.ravel(), bins=bins, **kwargs)
    print(h, binedges)
    b = np.repeat(binedges, 2)[1:-1]
    h = np.repeat(h, 2)
    mplt.quick_scatter([b], [h], outf=outf)
