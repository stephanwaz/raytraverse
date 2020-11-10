# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import subprocess

import numpy as np
import raytraverse
from raytraverse import translate, io, skycalc
from raytraverse.mapper import ViewMapper
from raytraverse.crenderer import cRtrace
from raytraverse.integrator.metricset import MetricSet


class BaseIntegrator(object):
    """base integrator class, can be used on StaticFields

    This class provides an interface to:

        1. output luminance maps as hdr images, photometric quantities, and
           visual comfort metrics for a static (single sky condition lightfield

    Parameters
    ----------
    lightfield: raytraverse.lightfield.LightFieldKD
        class containing sample data
    """
    rowheaders0 = dict(idx=["pt-idx"], sensor=["x", "y", "z", "dx", "dy", "dz"])
    rowheaders1 = dict(err=["pt-err"])

    def __init__(self, lightfield):
        #: raytraverse.lightfield.LightFieldKD
        self.skyfield = lightfield
        #: raytraverse.scene.Scene
        self.scene = lightfield.scene
        self.scene.log(self, "Initializing")
        self.metricreturn = None
        self.metricheader = None

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

    def header(self):
        """generate image header string"""
        octe = f"{self.scene.outdir}/scene.oct"
        hdr = subprocess.run(f'getinfo {octe}'.split(), capture_output=True,
                             text=True)
        hdr = [i.strip() for i in hdr.stdout.split('\n')]
        hdr = [i for i in hdr if i[0:5] == 'oconv']
        tf = "%Y:%m:%d %H:%M:%S"
        hdr.append("CAPDATE= " + datetime.now().strftime(tf))
        hdr.append("GMT= " + datetime.now(timezone.utc).strftime(tf))
        hdr.append(f"SOFTWARE= {cRtrace.version}")
        lastmod = os.path.getmtime(os.path.dirname(raytraverse.__file__))
        tf = "%a %b %d %H:%M:%S %Z %Y"
        lm = datetime.fromtimestamp(lastmod, timezone.utc).strftime(tf)
        hdr.append(
            f"SOFTWARE= RAYTRAVERSE {raytraverse.__version__} lastmod {lm}")
        return hdr

    def hdr(self, pi, vm, pdirs, mask, vstr, outf, interp=1, altlf=None,
            coefs=1):
        """interpolate and write hdr image for a single skyv/sun/pt-index
        combination

        Parameters
        ----------
        pi: int, tuple
            index value in lightfield
        vm: raytraverse.mapper.viewmapper
            should have a view angle of 180 degrees, the analyis direction
        pdirs: np.array
            pixel ray directions, shape (res, res, 3)
        mask: tuple
            pair of integer np.array representing pixel coordinates of images
            to calculate
        vstr: str
            view string for radiance compatible header
        outf: str
            destination file path
        interp: int
            number of rays to search for in query, interpolation always happens
            between 3 points, but in order to find a valid mesh triangle more
            than 3 points is typically necessary. 16 seems to be a safe number
            set to 1 (default) to turn off interpolation and use nearest ray
            this will result in voronoi patches in the final image.
        altlf: raytraverse.lightfield.Lightfield
            substitute lightfield to use instead of self.skyfield
        coefs: np.array
            passed to lightfield.add_to_image
        """
        img = np.zeros(pdirs.shape[:-1])
        if altlf is None:
            lf = self.skyfield
        else:
            lf = altlf
        try:
            lf.add_to_img(img, mask, pi, pdirs[mask], coefs=coefs, vm=vm,
                          interp=interp)
        except KeyError:
            print(f"skipped (no entry in LightField): {outf}", file=sys.stderr)
        else:
            io.array2hdr(img, outf, self.header() + [vstr])
            print(outf, file=sys.stderr)

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
            data = np.concatenate((info[0], fmetric, info[1]))
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

    def _all_hdr(self, exc, pi, pj, vdir, res=400, viewangle=180.0,
                 vname='view', interp=1, altlf=None):
        """handler for making calls to hdr submitted to a ProcessPoolExecutor

        overrides must match return type, unless integrate is also overriden

        Parameters
        ----------
        exc: ProcessPoolExectutor()
        pi: int
            point index in lightfiled
        vdir:
            (dx, dy, dz) view direction
        res: int, optional
            image resolution
        viewangle: float, optional
            1-180, opening angle of angular fisheye
        vname: str, optional
            incorporated into out file naming
        interp: int, optional
            if greater than 1 bandwidth for finding interpolation triangle
            See Also raytraverse.lightfield.LightFieldKD.interpolate_query()
        altlf: raytraverse.lightfield.LightFieldKD
            if none defaults to self.skyfielc

        Returns
        -------
        list
            of ConcurrentFutures.future objects
        """
        vm = ViewMapper(viewangle=viewangle, dxyz=vdir, name=vname)
        vp = self.scene.area.idx2pt([pi])[0]
        vstr = ('VIEW= -vta -vv {0} -vh {0} -vd {1} {2} {3}'
                ' -vp {4} {5} {6}'.format(viewangle, *vdir, *vp))
        pdirs = vm.pixelrays(res)
        mask = vm.in_view(pdirs)
        outf = (f"{self.scene.outdir}_"
                f"{vname}_{pj:04d}_{altlf.prefix}.hdr")
        return [exc.submit(self.hdr, pi, vm, pdirs, mask, vstr, outf, interp,
                           altlf)]

    def _all_metric(self, exc, pi, vdir, pt, perr, metricset,
                    altlf, **kwargs):
        """handler for making calls to metric submitted to a ProcessPoolExecutor

        overrides must match return type, unless integrate is also overriden

        Parameters
        ----------
        exc: ProcessPoolExectutor()
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
        return [exc.submit(self.metric, pi, vm, metricset, info, altlf,
                           **kwargs)]

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
        with ProcessPoolExecutor(io.get_nproc()) as exc:
            fu = []
            last_pi = None
            ptlf = None
            loopdat = zip(s_pis, s_pts[:, 0:3], s_pts[:, 3:6], s_perrs)
            for pj, (pi, pt, vdir, perr) in enumerate(loopdat):
                if pi != last_pi:
                    ptlf = self.pt_field(pi)
                    last_pi = pi
                if dohdr:
                    fu += self._all_hdr(exc, pi, pj, vdir, res=res,
                                        viewangle=viewangle, vname=vname,
                                        interp=interp, altlf=ptlf)
                if dometric:
                    fu += self._all_metric(exc, pi, vdir, pt, perr, metricset,
                                           ptlf, **kwargs)
            outmetrics = []
            for future in fu:
                out = future.result()
                if out is not None:
                    outmetrics.append(out)
        if dometric:
            d = np.array(outmetrics)
            cols = d.shape[1]
            unsort = np.argsort(sort)
            d = d.reshape((len(pts), -1, cols))[unsort].reshape(-1, cols)
            return d
        return np.array([])
