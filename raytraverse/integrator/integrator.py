# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import sys

import numpy as np

from raytraverse import translate
from raytraverse.evaluate import MetricSet
from raytraverse.lightfield.lightresult import ResultAxis, LightResult
import raytraverse.integrator._helpers as intg
from raytraverse.mapper import ViewMapper
from raytraverse.utility import pool_call


class Integrator(object):
    """collection of lightplanes with KDtree structure for sun position query

    Parameters
    ----------
    lightplanes: Sequence[raytraverse.lightfield.LightPlaneKD]
    """
    evaluate_pt = intg.evaluate_pt
    img_pt = intg.img_pt

    def __init__(self, *lightplanes, includesky=True, includesun=True,
                 sunviewengine=None):
        self.scene = lightplanes[0].scene
        self.lightplanes = lightplanes
        self.includesky = includesky
        self.includesun = includesun
        self._sunviewengine = sunviewengine

    def make_images(self, skydata, points, vm, viewangle=180., res=512,
                    interp=False, prefix="img", namebyindex=False, suntol=10.0,
                    blursun=False, resamprad=0.0):
        """see namebyindex for file naming conventions

        Parameters
        ----------
        skydata: raytraverse.sky.Skydata
        points: np.array
            shape (N, 3)
        vm: Union[raytraverse.mapper.ViewMapper, np.array]
            either a predefined ViewMapper (used for all points) or an array
            of view directions (will use a 180 degree view angle when
            initializing ViewMapper)
        viewangle: float, optional
            view opening for sensor (0-180,360) when vm is given as an array
            of view directions.
        res: int, optional
            image resolution
        interp: bool, optional
            interpolate image
        prefix: str, optional
            prefix for output file naming
        namebyindex: bool, optional
            if False (default), names images by:
            <prefix>_sky-<row>_pt-<x>_<y>_<z>_vd-<dx>_<dy>_<dz>.hdr
            if True, names images by:
            <prefix>_sky-<row>_pt-<pidx>_vd-<vidx>.hdr, where pidx, vidx are
            refer to the order of points, and vm.

        Returns
        -------
        np.array of out_files shape (skies, points, views)

        """
        points = np.atleast_2d(points)
        if hasattr(vm, "dxyz"):
            vms = [vm]
        else:
            dxyz = np.asarray(vm).reshape(-1, 3)
            vms = [ViewMapper(d, viewangle) for d in dxyz]
        oshape = (len(skydata.maskindices), len(points), len(vms))
        idxs, skydatas, dsns, vecs = self._group_query(skydata, points)
        self.scene.log(self, f"Making Images for {oshape[2]}"
                             f" view directions at {oshape[1]} points under "
                             f"{oshape[0]} skies", True)
        skylabs = [f"sky-{si:04d}" for si in skydata.maskindices]
        if namebyindex:
            ptlabs = [f"pt-{i:03d}" for i in range(len(points))]
            viewlabs = [f"vd-{i:03d}" for i in range(len(vms))]
        else:
            ptlabs = ["pt-{:.1f}_{:.1f}_{:.1f}".format(*pt) for pt in points]
            viewlabs = ["vd-{:.1f}_{:.1f}_{:.1f}".format(*v.dxyz) for v in vms]
        a, b, c = np.meshgrid(skylabs, ptlabs,
                              viewlabs, indexing='ij')
        combos = np.stack((a.ravel().astype(object), b.ravel(), c.ravel()))
        combos = combos.T.reshape(-1, len(vms), 3)

        # use parallel processing
        skinfo = skydata.skydata[skydata.fullmask]
        s = (*oshape[0:2], skinfo.shape[1])
        skinfo = np.broadcast_to(skinfo[:, None, :], s).reshape(-1, s[-1])
        mask_kwargs = dict(combos=combos, qpts=vecs[:, 3:], skinfo=skinfo)
        outfs, _ = self._process_mgr(idxs, skydatas, dsns, type(self).img_pt,
                                     message="Making Images",
                                     mask_kwargs=mask_kwargs, vms=vms, res=res,
                                     interp=interp, prefix=prefix,
                                     suntol=suntol, blursun=blursun,
                                     resamprad=resamprad)
        return sorted([i for j in outfs for i in j])

    def evaluate(self, skydata, points, vm, viewangle=180.,
                 metricclass=MetricSet, metrics=None, datainfo=False,
                 srconly=False, suntol=10.0, blursun=False, coercesumsafe=False,
                 **kwargs):
        """apply sky data and view queries to daylightplane to return metrics
        parallelizes and optimizes run order.

        Parameters
        ----------
        skydata: raytraverse.sky.Skydata
        points: np.array
            shape (N, 3)
        vm: Union[raytraverse.mapper.ViewMapper, np.array]
            either a predefined ViewMapper (used for all points) or an array
            of view directions (will use 'viewangle' when
            initializing ViewMapper)
        viewangle: float, optional
            view opening for sensor (0-180,360) when vm is given as an array
            of view directions, note that for illuminance based
            metrics, a value of 360 may not make sense as values
            behind will be negative.
        metricclass: raytraverse.evaluate.BaseMetricSet, optional
        metrics: Sized, optional
        srconly: bool, optional
            sun only calculations
        suntol: float, optional
            if Integrator has an engine, resample sun views when actual sun
            position error is greater than this many degrees.
        blursun: bool, optional
            apply human PSF to small bright sources
        coercesumsafe: bool, optional
            attempt to calculate sumsafe metrics
        datainfo: Union[Sized[str], bool], optional
            include information about source data as additional metrics. Valid
            values include: ["pt_err", "pt_idx", "src_err", "src_idx"].
            If True, includes all.

        Returns
        -------
        raytraverse.lightfield.LightResult
        """
        points = np.atleast_2d(points)
        (vm, vms, cmetrics, ometrics,
         sumsafe, needs_post) = self._check_params(vm, viewangle, metrics,
                                                   metricclass, coercesumsafe)
        tidxs, skydatas, dsns, vecs = self._group_query(skydata, points)
        oshape = (len(skydata.maskindices), len(points), len(vms), len(cmetrics))
        self.scene.log(self, f"Evaluating {oshape[3]} metrics for {oshape[2]}"
                             f" view directions at {oshape[1]} points under "
                             f"{oshape[0]} skies", True)
        fields, isort = self._process_mgr(tidxs, skydatas, dsns,
                                          type(self).evaluate_pt,
                                          message="Evaluating Points",
                                          srconly=srconly, sumsafe=sumsafe,
                                          metricclass=metricclass,
                                          metrics=cmetrics, vm=vm, vms=vms,
                                          suntol=suntol, blursun=blursun,
                                          **kwargs)
        # sort back to original order and reshape
        fields = np.concatenate(fields, axis=0)[isort].reshape(oshape)
        if needs_post:
            ofields = np.zeros(fields.shape[:-1] + (len(ometrics),))
            for i, m in enumerate(ometrics):
                if m in cmetrics:
                    ofields[..., i] = fields[..., cmetrics.index(m)]
                elif m == "dgp":
                    illum = fields[..., cmetrics.index("illum")]
                    pwsl2 = fields[..., cmetrics.index("pwsl2")]
                    t1 = 5.87 * 10**-5 * illum
                    t2 = 9.18 * 10**-2 * np.log10(1 + pwsl2/np.power(illum, 1.87))
                    ll = 1
                    if "lowlight" in kwargs and kwargs["lowlight"]:
                        ll = np.where(illum < 500, np.exp(0.024*illum - 4) /
                                      (1 + np.exp(0.024*illum - 4)), 1)
                    ofields[..., i] = np.minimum(ll*(t1 + t2 + 0.16), 1.0)
                elif m == "ugp":
                    pwsl2 = fields[..., cmetrics.index("pwsl2")]
                    backlum = fields[..., cmetrics.index("backlum")]
                    with np.errstate(divide='ignore'):
                        ugr = np.maximum(0, 8*np.log10( 0.25*pwsl2/backlum))
                    ofields[..., i] = (1 + 2/7*10**(-(ugr + 5)/40))**-10
            fields = ofields
        sinfo, dinfo = self._sinfo(datainfo, vecs, tidxs, oshape[0:2])
        if sinfo is not None:
            nshape = list(sinfo.shape)
            nshape[2] = fields.shape[2]
            sinfo = np.broadcast_to(sinfo, nshape)
            fields = np.concatenate((fields, sinfo), axis=-1)
        # compose axes: (skyaxis, ptaxis, viewaxis, metricaxis)
        axes = (ResultAxis(skydata.rowlabel[skydata.fullmask], f"sky"),
                ResultAxis(points, "point"),
                ResultAxis([v.dxyz for v in vms], "view"),
                ResultAxis(list(ometrics) + dinfo, "metric"))
        lr = LightResult(fields, *axes)
        return lr

    @staticmethod
    def _check_params(vm, viewangle=180., metrics=None, metricclass=MetricSet,
                      coercesumsafe=False):
        if hasattr(vm, "dxyz"):
            vms = [vm]
        else:
            dxyz = np.asarray(vm).reshape(-1, 3)
            vm = ViewMapper()
            vms = [ViewMapper(d, viewangle) for d in dxyz]
        if metrics is None:
            metrics = metricclass.defaultmetrics
        needs_post = False
        if coercesumsafe:
            ometrics = []
            cmetrics = set()
            s2s = True
            for m in metrics:
                if m in metricclass.safe2sum:
                    cmetrics.add(m)
                    ometrics.append(m)
                elif m == "dgp":
                    cmetrics.update(["illum", "pwsl2"])
                    ometrics.append(m)
                    needs_post = True
                elif m == "ugp":
                    cmetrics.update(["backlum", "pwsl2"])
                    ometrics.append(m)
                else:
                    print(f"could not coerce metric {m} to sumsafe",
                          file=sys.stderr)
            cmetrics = list(cmetrics)
        else:
            ometrics = metrics
            cmetrics = metrics
            s2s = metricclass.check_safe2sum(metrics)
        return vm, vms, cmetrics, ometrics, s2s, needs_post

    def _group_query(self, skydata, points):
        # query and group sun positions
        gshape = (len(skydata.maskindices), len(points))
        suns = skydata.sun[:, None, 0:3]
        pts = np.atleast_2d(points)[None, :]
        vecs = np.concatenate(np.broadcast_arrays(suns, pts), -1).reshape(-1, 6)
        skydatas, dsns = self._unroll_sky_grid(skydata, gshape)
        idxs = []
        for lp in self.lightplanes:
            if lp.vecs.shape[1] == 6:
                idxs.append(lp.query(vecs)[0])
            else:
                idx, err = lp.query(points)
                idxs.append(np.broadcast_to(idx, gshape).ravel())
        return np.stack(idxs), skydatas, dsns, vecs

    def _sinfo(self, datainfo, vecs, idxs, oshape):
        """error and bin information for evaluate queries"""
        dinfo = ["src_err", "src_idx", "pt_err", "pt_idx"]
        if not datainfo:
            dinfo = []
        elif hasattr(datainfo, "__len__"):
            dinfo = [i for i in dinfo if i in datainfo]
        if len(dinfo) == 0:
            return None, dinfo
        dinfo2 = []
        sinfo = []
        for lp, idx in zip(self.lightplanes, idxs):
            for di in dinfo:
                if di == "src_err" and lp.vecs.shape[1] == 6:
                    snerr = np.linalg.norm(vecs[:, :3] - lp.vecs[idx, :3],
                                           axis=-1)
                    sinfo.append(translate.chord2theta(snerr)*(180/np.pi))
                    dinfo2.append(f"{lp.src}_{di}")
                elif di == "src_idx" and lp.vecs.shape[1] == 6:
                    srcidx = lp.data.idx[idx, 0]
                    sinfo.append(srcidx)
                    dinfo2.append(f"{lp.src}_{di}")
                elif di == "pt_err":
                    pterr = np.linalg.norm(vecs[:, 3:] - lp.vecs[idx, -3:],
                                           axis=-1)
                    sinfo.append(pterr)
                    dinfo2.append(f"{lp.src}_{di}")
                elif di == "pt_idx":
                    try:
                        ptidx = lp.data.idx[idx, 1]
                    except IndexError:
                        ptidx = idx
                    sinfo.append(ptidx)
                    dinfo2.append(f"{lp.src}_{di}")
        return np.array(sinfo).T.reshape(*oshape, 1, -1), dinfo2

    def _unroll_sky_grid(self, skydata, oshape):
        # broadcast skydata to match full indexing
        s = (*oshape, skydata.smtx.shape[1])
        smtx = np.broadcast_to(skydata.smtx[:, None, :], s).reshape(-1, s[-1])
        ds = (*oshape, skydata.sun.shape[1])
        dsns = np.broadcast_to(skydata.sun[:, None, :], ds).reshape(-1, ds[-1])

        skydatas = []
        # generic, assumes sun and/or sky, uses patch sun when no sun present
        skyi = None
        suni = None
        for i, lp in enumerate(self.lightplanes):
            if lp.vecs.shape[1] == 3:
                skyi = i
                skydatas.append(smtx)
            elif self.includesun:
                suni = i
                skydatas.append(dsns[:, 3:4])
        if self.includesun and suni is None and skyi is not None:
            dprx = skydata.smtx_patch_sun(includesky=self.includesky)
            dprx = np.broadcast_to(dprx[:, None, :], s).reshape(-1, s[-1])
            skydatas[skyi] = dprx
        return skydatas, dsns

    @staticmethod
    def _sort_run_data(tidxs):
        # unique returns sorted values (evaluation order)
        qtup, qidx = np.unique(tidxs.T, axis=0, return_index=True)
        tup_sort = np.lexsort(tidxs[::-1])
        # make an inverse sort to undo evaluation order
        tup_isort = np.argsort(tup_sort, kind='stable')
        return qtup, qidx, tup_isort

    def _process_mgr(self, tidxs, skydatas, dsns, eval_fn,
                     message="Evaluating Points", mask_kwargs=None,
                     **eval_kwargs):

        qtup, qidx, tup_isort = self._sort_run_data(tidxs)

        d = np.linspace(0, len(qtup), max(2, int(len(qtup)/250))).astype(int)
        slices = [slice(d[i], d[i + 1], 1) for i in range(len(d) - 1)]
        self.scene.log(self, f"Calculating {len(qidx)} sun/sky/pt combinations",
                       True)
        if self._sunviewengine is not None:
            eval_kwargs.update(svengine=self._sunviewengine)
        fields = []
        for i, slc in enumerate(slices):
            lptis = []
            for qi, qt in zip(qidx[slc], qtup[slc]):
                lptis.append(([(lp.data, tidx[qi]) for lp, tidx
                               in zip(self.lightplanes, tidxs)], qt))
            desc = f"{message} ({i+1:02d} of {len(slices):02d})"
            fields += pool_call(_load_pts, lptis, eval_fn, tidxs.T, mask_kwargs,
                                skydatas, dsns, desc=desc, pbar=self.scene.dolog, **eval_kwargs)
        return fields, tup_isort


def _load_pts(lpti, qt, eval_fn, idx_tup, mask_kwargs, skydatas, dsns,
              **kwargs):
    lpts = []
    sx = []
    mask = np.all(idx_tup == qt, -1)
    for (lp, tidx), smx in zip(lpti, skydatas):
        if tidx >= 0:
            lpts.append(lp[tidx])
            sx.append(smx[mask])
    if mask_kwargs is not None:
        for k, v in mask_kwargs.items():
            kwargs.update([(k, v[mask])])
    return eval_fn(lpts, sx, dsns[mask], **kwargs)
