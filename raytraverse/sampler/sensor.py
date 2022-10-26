# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import translate


filterdict = {
              'wav': (np.array([[-1, 2, -1]])/2, np.array([[-1], [2], [-1]])/2,
                      np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]])/2),
              'haar': (np.array([[1, -1]])/2, np.array([[1], [-1]])/2,
                       np.array([[1, 0], [0, -1]])/2)
              }


class Sensor(object):
    """for use as engine in area sampler, holds collection of multiple sensor
    directions and offsets

    Parameters
    ----------
    engine: craytraverse.renderer.Rennderer
        fully initialized renderer class instance
    dirs: Sequence, optional
        array like shape (N, 3) sensor directions
    offsets: Sequence, optional
        array like shape (N, 3) offsets from sample position to include
        (for example mulitple z-heights)
    sunview: bool, optional
        NOT IMPLEMENTED
        if True, dirs are treated as candidate reflection normals, a value
        of (0, 0, 0) is prepended to hold the direct view.
    """

    def __init__(self, engine, dirs=(0.0, 0.0, 1.0), offsets=(0.0, 0.0, 0.0),
                 name="sensor", sunview=False):
        self.name = name
        self.sunview = sunview
        self.engine = engine
        self.dirs = translate.norm(np.atleast_2d(dirs))
        # if sunview:
        #     self.dirs = np.concatenate((((0, 0, 0),), self.dirs))
        self.offsets = np.atleast_2d(offsets)
        d, o = np.broadcast_arrays(self.dirs[None], self.offsets[:, None])
        self.sensors = np.hstack((o.reshape(-1, 3), d.reshape(-1, 3)))
        self.features = self.sensors.shape[0]

    @property
    def nproc(self):
        return self.engine.nproc

    def __call__(self, rays):
        srays = np.copy(self.stack_rays(rays).reshape(-1, 6), 'C')
        r = self.engine.run(srays)
        return r.reshape(len(rays), len(self.sensors), *r.shape[1:])

    def run(self, *args, **kwargs):
        """alias for call, for consistency with SamplerPt classes for nested
        dimensions of evaluation"""
        return self(args[0])

    def stack_rays(self, r):
        r = np.atleast_2d(r)
        srays = np.hstack((r, np.zeros(r.shape)))
        return srays[:, None] + self.sensors[None]
