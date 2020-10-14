# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
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


class StaticIntegrator(object):
    """class to generate outputs from lightfield(s) and sky conditions

    This class provides an interface to:

        1. output luminance maps as hdr images, photometric quantities, and
           visual comfort metrics for a static (single sky condition lightfield

    Parameters
    ----------
    lightfield: raytraverse.lightfield.LightFieldKD
        class containing sample data
    """

    def __init__(self, lightfield):
        #: raytraverse.lightfield.LightFieldKD
        self.skyfield = lightfield
        #: raytraverse.scene.Scene
        self.scene = lightfield.scene

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

    def hdr(self, pi, vm, pdirs, mask, vstr, outf, interp=1):
        """interpolate and write hdr image for a single skyv/sun/pt-index
        combination

        Parameters
        ----------
        pi: int
            point index
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

        Returns
        -------
        outf: str
            saved output file
        returntype: str
            'hdr' indicating format of result (useful when called
            as parallel process to seperate from 'metric' or other outputs)

        """
        img = np.zeros(pdirs.shape[:-1])
        self.skyfield.add_to_img(img, mask, pi, pdirs[mask], vm=vm,
                                 interp=interp)
        io.array2hdr(img, outf, self.header() + [vstr])
        return outf, 'hdr'

    def metric(self, pi, vm, metricset, info, **kwargs):
        """calculate metrics for a single skyv/sun/pt-index combination

        Parameters
        ----------
        pi: int
            point index
        vm: raytraverse.mapper.ViewMapper
            analysis point
        metricset: str, list
            string or list of strings naming metrics.
            see raytraverse.integrator.MetricSet.metricdict for valid names
        info: list
            constant column values to include in row output
        kwargs:
            passed to metricset

        Returns
        -------
        data: np.array
            results for skyv and pi, shape (len(info[0]) + len(metricfuncs) +
            1 + len(info[1])
        returntype: str
            'metric' indicating format of result (useful when called
            as parallel process to seperate from 'hdr' or other outputs)
        """
        rays, omega, lum = self.skyfield.get_applied_rays(pi, vm, 1.0)
        fmetric = MetricSet(vm, rays, omega, lum, metricset, **kwargs)()
        data = np.concatenate((info[0], fmetric, info[1]))
        return data, 'metric'

    @staticmethod
    def _metric_info(metricreturn, pi, sensor, perr):
        info = [[], []]
        infos0 = dict(idx=[pi], sensor=sensor)
        infos1 = dict(err=[perr])
        for k, v in infos0.items():
            if metricreturn[k]:
                info[0] += v
        for k, v in infos1.items():
            if metricreturn[k]:
                info[1] += v
        return info

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
        header: list
            if dometric, a list of column headers, else an empty list
        data: np.array
            if dometric, an array of output data, else None

        """
        metricreturn = {"idx": False, "sensor": False,
                        "err": False}
        if metric_return_opts is not None:
            metricreturn.update(metric_return_opts)
        metricreturn["metric"] = True
        perrs, pis = self.scene.area.pt_kd.query(pts[:, 0:3])
        sort = np.argsort(pis)
        s_pis = pis[sort]
        s_perrs = perrs[sort]
        s_pts = pts[sort]
        with ProcessPoolExecutor(io.get_nproc()) as exc:
            fu = []
            loopdat = zip(s_pis, s_pts[:, 0:3], s_pts[:, 3:6], s_perrs)
            for pj, (pi, pt, vdir, perr) in enumerate(loopdat):
                if dohdr:
                    vm = ViewMapper(viewangle=viewangle, dxyz=vdir, name=vname)
                    vp = self.scene.area.idx2pt([pi])[0]
                    vstr = ('VIEW= -vta -vv {0} -vh {0} -vd {1} {2} {3}'
                            ' -vp {4} {5} {6}'.format(viewangle, *vdir, *vp))
                    pdirs = vm.pixelrays(res)
                    mask = vm.in_view(pdirs)
                    outf = (f"{self.scene.outdir}_"
                            f"{vname}_{pj:04d}_{self.skyfield.prefix}.hdr")
                    fu.append(exc.submit(self.hdr, pi, vm, pdirs,
                                         mask, vstr, outf, interp))
                if dometric:
                    vm = ViewMapper(viewangle=180, dxyz=vdir)
                    info = self._metric_info(metricreturn, pi, [*pt, *vdir],
                                             perr)
                    fu.append(exc.submit(self.metric, pi, vm, metricset, info,
                                         **kwargs))
            outmetrics = []
            for future in fu:
                out, kind = future.result()
                if kind == 'hdr':
                    print(out, file=sys.stderr)
                elif kind == 'metric':
                    outmetrics.append(out)
        if dometric:
            if len(metricset) == 0:
                metricset = MetricSet.allmetrics
            headers = dict(
                idx=["pt-idx"],
                sensor=["x", "y", "z", "dx", "dy", "dz"],
                metric=(" ".join(metricset)).split(),
                err=["pt-err"])
            colhdr = []
            for k, v in headers.items():
                if metricreturn[k]:
                    colhdr += v
            d = np.array(outmetrics)
            cols = d.shape[1]
            unsort = np.argsort(sort)
            d = d.reshape((len(pts), -1, cols))[unsort].reshape(-1, cols)
            return colhdr, d
        return None, []
