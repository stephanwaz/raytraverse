# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from raytraverse.mapper import ViewMapper
from raytraverse.evaluate.metricset import MetricSet


class Metric(object):
    """metric calculator

    This class provides an interface to:

        1. output luminance maps as hdr images, photometric quantities, and
           visual comfort metrics for a static (single sky condition lightfield

    Parameters
    ----------
    scene: raytraverse.scene.Scene
    """
    rowheaders0 = dict(idx=["pt-idx", "sky-idx"],
                       sensor=["x", "y", "z", "dx", "dy", "dz"])
    rowheaders1 = dict(err=["pt-err", "sun-err"],
                       sky=["sun-x", "sun-y", "sun-z", "dir-norm",
                            "diff-horiz"])

    def __init__(self, scene, metricset=None, metric_return_opts=None):
        #: raytraverse.scene.Scene
        self.scene = scene
        self.metricreturn = metric_return_opts
        self.metricheader = metricset

    @property
    def metricreturn(self):
        """dictionary of booleans controlling metric return output columns
        and header generation"""
        return self._metricreturn

    @metricreturn.setter
    def metricreturn(self, metric_return_opts):
        self._metricreturn = {"idx": False, "sensor": False,
                              "err": False, "sky": False}
        if metric_return_opts is not None:
            self._metricreturn.update(metric_return_opts)
        self._metricreturn["metric"] = True

    @property
    def metricheader(self):
        """list of column names for returning with metric data"""
        return self._metricheader

    @metricheader.setter
    def metricheader(self, metricset):
        if metricset is None or len(metricset) == 0:
            metricset = MetricSet.defaultmetrics
        self._metricheader = []
        for k, v in self.rowheaders0.items():
            if self.metricreturn[k]:
                self._metricheader += v
        self._metricheader += (" ".join(metricset)).split()
        for k, v in self.rowheaders1.items():
            if self.metricreturn[k]:
                self._metricheader += v

    def metric(self, pi, vm, metricset, info, altlf=None, coefs=1.0,
               sunvec=None, **kwargs):
        """calculate metrics for a single skyv/sun/pt-index combination

        Parameters
        ----------
        pi: int, tuple
            index value in lightfield
        vm: raytraverse.mapper.ViewMapper
            analysis point
        metricset: str, list
            string or list of strings naming metrics.
            see raytraverse.integrator.MetricSet.allmetrics for valid names
        info: list
            constant column values to include in row output
        altlf: raytraverse.lightfield.Lightfield
            substitute lightfield to use instead of self.skyfield
        kwargs:
            passed to metricset

        Returns
        -------
        data: np.array
            results for skyv and pi, shape (len(info[0]) + len(metricfuncs) +
            1 + len(info[1])
        """
        if altlf is None:
            lf = self.skyfield
        else:
            lf = altlf
        try:
            rays, omega, lum = lf.get_applied_rays(pi, vm, coefs, sunvec=sunvec)
        except KeyError:
            print(f"skipped (no entry in LightField): {pi}, returning zeros")
            data = np.zeros(len(info[0]) + len(info[1]) +
                            len(self.metricheader))
        else:
            fmetric = MetricSet(vm, rays, omega, lum, metricset, **kwargs,
                                **lf.size)()
            data = np.nan_to_num(np.concatenate((info[0], fmetric, info[1])))
        return data

    def _metric_info(self, pi, sensor, perr, sky=None):
        """prepares information to concatenate with metric results"""
        info = [[], []]
        infos0 = dict(idx=pi, sensor=sensor)
        infos1 = dict(err=perr)
        if sky is not None:
            infos1['sky'] = [*sky]
        for k, v in infos0.items():
            if self.metricreturn[k]:
                info[0] += v
        for k, v in infos1.items():
            if self.metricreturn[k]:
                info[1] += v
        return info

    def _all_metric(self, pi, vdir, pt, perr, metricset,
                    altlf, **kwargs):
        """handler for making calls to metric submitted to a ProcessPoolExecutor

        overrides must match return type, unless integrate is also overriden

        Parameters
        ----------
        pi: int
            point index in lightfiled
        vdir:
            (dx, dy, dz) view direction
        pt:
            (x, y, z) view point
        perr: float
            euclidean distance from queried point to point in lightfield
        metricset: list
            keys of metrics to return (see raytraverse.integrator.MetricSet)
        altlf: raytraverse.lightfield.LightFieldKD
            if none defaults to self.skyfielc
        kwargs:
            passed to self.metric

        Returns
        -------
        list
            of ConcurrentFutures.future objects

        """
        vm = ViewMapper(viewangle=180, dxyz=vdir)
        info = self._metric_info([pi], [*pt, *vdir], [perr])
        return [self.metric(pi, vm, metricset, info, altlf, **kwargs)]

    def pt_field(self, pi):
        """lightfield to use for a particular point index"""
        return self.skyfield

    def integrate(self, pts, dohdr=True, dometric=True, vname='view',
                  viewangle=180.0, res=400, interp=1, metricset="illum",
                  metric_return_opts=None, **kwargs):
        """iterate through points and sky vectors to efficiently compute
        both hdr output and metrics, sharing intermediate calculations where
        possible.

        Parameters
        ----------
        pts: np.array
            shape (N, 6) points (with directions) to compute
        dohdr: bool, optional
            if True, output hdr images
        dometric: bool, optional
            if True, output metric data
        vname: str, optional
            view name for hdr output
        viewangle: float, optional
            view angle (in degrees) for hdr output (always an angular fisheye)
        res: int, optional
            pixel resolution of output hdr
        interp: int, optional
            if greater than one the bandwidth to search for nearby rays. from
            this set, a triangle including the closest point is formed for a
            barycentric interpolation.
        metricset: str, list, optional
            string or list of strings naming metrics.
            see raytraverse.integrator.MetricSet.allmetrics for valid choices
        metric_return_opts: dict, optional
            boolean dictionary of columns to print with metric output. Default:
            {"idx": False, "sensor": False, False, "sky": False}
        kwargs:
            additional parameters for integrator.MetricSet

        Returns
        -------
        data: np.array
            if dometric, an array of output data, else and empty list

        """
        output = " and ".join([i for i, j in (("images", dohdr),
                                              ("metrics", dometric)) if j])
        self.scene.log(self, f"integrating {output}")
        self.metricreturn = metric_return_opts
        self.metricheader = metricset
        perrs, pis = self.scene.area.pt_kd.query(pts[:, 0:3])
        sort = np.argsort(pis)
        s_pis = pis[sort]
        s_perrs = perrs[sort]
        s_pts = pts[sort]
        # with ProcessPoolExecutor(io.get_nproc()) as exc:
        fu = []
        last_pi = None
        ptlf = None
        loopdat = zip(s_pis, s_pts[:, 0:3], s_pts[:, 3:6], s_perrs)
        for pj, (pi, pt, vdir, perr) in enumerate(loopdat):
            if pi != last_pi:
                ptlf = self.pt_field(pi)
                last_pi = pi
            if dohdr:
                fu += self._all_hdr(pi, pj, vdir, res=res,
                                    viewangle=viewangle, vname=vname,
                                    interp=interp, altlf=ptlf)
            if dometric:
                fu += self._all_metric(pi, vdir, pt, perr, metricset, ptlf,
                                       **kwargs)
        outmetrics = []
        for future in fu:
            out = future
            if out is not None:
                outmetrics.append(out)
        if dometric:
            d = np.array(outmetrics)
            cols = d.shape[1]
            unsort = np.argsort(sort)
            d = d.reshape((len(pts), -1, cols))[unsort].reshape(-1, cols)
            return d
        return np.array([])
