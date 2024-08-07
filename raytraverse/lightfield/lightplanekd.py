# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from raytools import io
from raytraverse.evaluate import MetricSet
from raytraverse.lightfield.sets import LightPointSet
from raytraverse.lightfield.lightfield import LightField
from raytraverse.lightpoint.lightpointkd import calc_voronoi_area


class LightPlaneKD(LightField):
    """collection of lightpoints with KDtree structure for positional query"""

    @property
    def data(self):
        """LightPointSet"""
        return self._data

    @ data.setter
    def data(self, idx):
        self._data = LightPointSet(self.scene, self.vecs, idx, self.src,
                                   self.pm.name)

    @property
    def omega(self):
        """representative area of each point

        :getter: Returns array of areas
        :setter: sets areas
        :type: np.array
        """
        if self._omega is None:
            self._omega = calc_voronoi_area(self.vecs, self.pm)
        return self._omega

    @omega.setter
    def omega(self, oga):
        """wait to calculate area until needed"""
        self._omega = None

    def evaluate(self, skyvec, points=None, vm=None, metricclass=MetricSet,
                 metrics=None, mask=True, **kwargs):
        # qidx are the unique query indices and midx are the mapping indices
        # to restore full results from the qidx results
        if points is None:
            qidx = midx = np.arange(len(self.vecs))
        else:
            ridx, d = self.query(points)
            if mask:
                if hasattr(self.pm, "mask"):
                    omask = self.pm.mask
                    self.pm.mask = True
                    ridx = ridx[self.pm.in_view(points, False)]
                    self.pm.mask = omask
                else:
                    ridx = ridx[self.pm.in_view(points, False)]
            qidx, midx = np.unique(ridx, return_inverse=True)
        results = []
        for qi in qidx:
            lp = self.data[qi]
            vol = lp.evaluate(skyvec, vm=vm)
            if vm is None:
                vm = lp.vm
            results.append(metricclass(*vol, vm, metricset=metrics,
                                       **kwargs)())
        return np.array(results)[midx]

    def make_image(self, outf, vals, res=1024, interp=False, showsample=False):
        """make an image from precomputed values for every point in LightPlane

        Parameters
        ----------
        outf: str
            the file to write
        vals: np.array
            shape (len(self.points),) the values computed for each point
        res: int, optional
            image resolution (the largest dimension
        interp: bool, optional
            apply linear interpolation, points outside convex hull of results
            fall back to nearest
        showsample: bool, optionaal
            color pixel at sample location red
        """
        img, vecs, mask, _, header = self.pm.init_img(res)
        if interp:
            xyp = vecs[mask]
            interp = LinearNDInterpolator(self.vecs[:, 0:2], vals,
                                          fill_value=-1)
            lum = interp(xyp[:, 0], xyp[:, 1])
            neg = lum < 0
            i, d = self.query(xyp[neg])
            lum[neg] = vals[i]
            img[mask] = lum
        else:
            i, d = self.query(vecs[mask])
            img[mask] = vals[i]
        if showsample:
            img = np.repeat(img[None, ...], 3, 0)
            img = self.pm.add_vecs_to_img(img, self.vecs,
                                          channels=(1, 0, 0))
            io.carray2hdr(img, outf, header)
        else:
            io.array2hdr(img, outf, header)

    def direct_view(self, res=512, showsample=True, vm=None, area=False,
                    metricclass=MetricSet, metrics=('avglum',), interp=False):
        """create a summary image of lightplane showing samples and areas"""
        if area:
            outf = self._datadir.replace("/", "_") + f"{self.src}_area.hdr"
            self.make_image(outf, self.omega, res=res, showsample=showsample,
                            interp=False)
        if metrics is not None:
            result = self.evaluate(1, vm=vm, metricclass=metricclass,
                                   metrics=metrics, scale=1).T
            for r, m in zip(result, metrics):
                outf = self._datadir.replace("/", "_") + f"{self.src}_{m}.hdr"
                self.make_image(outf, r, res=res, showsample=showsample,
                                interp=interp)
