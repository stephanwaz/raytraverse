# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""functions for plotting data"""
import warnings

import numpy as np
from clipt.plot import get_colors
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon


def save_img(fig, ax, outf, title=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()
        if title is not None:
            ax.set_title(title)
        fig.savefig(outf)


def imshow(im, figsize=(10, 10), outf=None, **kwargs):
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.imshow(im.T, origin='lower', **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    save_img(fig, ax, outf=outf)


def mk_img_setup(lums, bounds=None, figsize=(10, 10), ext=1):
    defparam = {
        'font.weight': 'ultralight',
        'font.family': 'sans-serif',
        'font.size': 15,
        'axes.linewidth': 0,
        }
    rcParams.update(defparam)
    if bounds is None:
        minl = np.min(lums)
        maxl = np.max(lums)
    else:
        minl = bounds[0]
        maxl = bounds[1]
        lums = np.clip(lums, minl, maxl)
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot()
    norm = Normalize(vmin=minl, vmax=maxl)
    lev = np.linspace(minl, maxl, 200)
    ext = np.asarray(ext).flatten()
    if ext.size == 1:
        ax.set(xlim=(-ext, ext), ylim=(-ext, ext))
    elif ext.size == 2:
        ax.set(xlim=ext, ylim=ext)
    else:
        ax.set(xlim=ext[0:2], ylim=ext[2:4])
    return lums, fig, ax, norm, lev


def set_ang_ticks(ax, ext):
    ax.tick_params(length=10, width=.5, direction='inout', pad=5)
    ticks = np.linspace(-ext, ext, 7)
    labs = np.round(np.linspace(-ext*180/np.pi,
                                ext*180/np.pi, 7)).astype(int)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs)


def colormap(colors, norm):
    cmap = get_colors(colors)
    cmap.set_norm(norm)
    return cmap


def plot_patches(ax, patches, patchargs=None):
    if patchargs is None:
        patchargs = {}
    for lum, p in patches:
        ax.add_patch(Polygon(p, closed=True, facecolor=lum, **patchargs))
