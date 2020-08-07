# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""functions for plotting data"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from clipt import mplt


def imshow(im, figsize=(10, 10), outf=None, **kwargs):
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


def hist(lums, bins='auto', outf=None, **kwargs):
    h, binedges = np.histogram(lums.ravel(), bins=bins, **kwargs)
    print(h, binedges, file=sys.stderr)
    b = np.repeat(binedges, 2)[1:-1]
    h = np.repeat(h, 2)
    mplt.quick_scatter([b], [h], outf=outf)
