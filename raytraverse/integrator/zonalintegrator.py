# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import translate
from raytraverse.evaluate import MetricSet
from raytraverse.integrator import Integrator
from raytraverse.lightfield import SunsPlaneKD, ZonalLightResult
from raytraverse.integrator._helpers import calc_omega
from raytraverse.lightfield.lightresult import ResultAxis, LightResult
from raytraverse.utility import pool_call


class ZonalIntegrator(Integrator):

    def __init__(self, *lightplanes, includesky=True, includesun=True,
                 sunviewengine=None):
        super(ZonalIntegrator, self).__init__(*lightplanes,
                                              includesky=includesky,
                                              includesun=includesun,
                                              sunviewengine=sunviewengine)
        self._issunplane = [i for i, lp in enumerate(self.lightplanes)
                            if type(lp) == SunsPlaneKD]

    def evaluate(self, skydata, pm, vm, viewangle=180.,
                 metricclass=MetricSet, metrics=None, srconly=False,
                 ptfilter=.25, stol=10,  minsun=1, datainfo=False,
                 **kwargs):
        """apply sky data and view queries to daylightplane to return metrics
        parallelizes and optimizes run order.

        Parameters
        ----------
        skydata: raytraverse.sky.Skydata
        pm: raytraverse.mapper.PlanMapper
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
        ptfilter: Union[float, int], optional
            minimum seperation for returned points
        stol: Union[float, int], optional
            maximum angle (in degrees) for matching sun vectors
        minsun: int, optional
            if atleast these many suns are not returned based on stol, directly
            query for this number of results (regardless of sun error)
        datainfo: Union[Sized[str], bool], optional
            include information about source data as additional metrics. Valid
            values include: ["src_err", "src_idx"]. If True, includes both.

        Returns
        -------
        raytraverse.lightfield.LightResultKD
        """
        vm, vms, metrics, sumsafe = self._check_params(vm, viewangle,
                                                       metrics,
                                                       metricclass)
        (tidxs, skydatas, dsns,
         vecs, serr, areas) = self._group_query(skydata, pm, ptfilter=ptfilter,
                                                stol=stol, minsun=minsun)
        self.scene.log(self, f"Evaluating {len(metrics)} metrics for {len(vms)}"
                             f" view directions across zone '{pm.name}' under "
                             f"{len(skydata.sun)} skies", True)
        fields, isort = self._process_mgr(tidxs, skydatas, dsns,
                                          type(self).evaluate_pt,
                                          message="Evaluating Zones",
                                          srconly=srconly, sumsafe=sumsafe,
                                          metricclass=metricclass,
                                          metrics=metrics, vm=vm, vms=vms,
                                          **kwargs)
        fields = np.concatenate(fields, axis=0)[isort]
        oshape = (len(fields), len(vms))
        pmetrics = ['x', 'y', 'z', 'area']
        if datainfo:
            sinfo, dinfo = self._sinfo(serr, vecs, tidxs, oshape + (2,))
            if sinfo is not None:
                fields = np.concatenate((sinfo, fields), axis=-1)
                pmetrics += dinfo

        areas = np.broadcast_to(areas[:, None, None], oshape + (1,))
        axes = [ResultAxis(skydata.maskindices, f"sky"),
                ResultAxis([pm.name], f"zone"),
                ResultAxis([v.dxyz for v in vms], "view"),
                ResultAxis(pmetrics + metrics, "metric")]

        cnts = [len(v) for v in vecs]
        strides = np.cumsum(cnts)[:-1]
        fl = np.array(cnts)
        if np.all(fl == fl[0]):
            axes[1] = ResultAxis(vecs[0][:, 3:], "point")
            axes[3] = ResultAxis(["area"] + metrics, "metric")
            fields = np.concatenate((areas, fields), axis=-1)
            fields = fields.reshape((-1, cnts[0]) + fields.shape[1:])
            lr = LightResult(fields, *axes)
        else:
            fvecs = np.concatenate(vecs, axis=0)
            fvecs = np.broadcast_to(fvecs[:, None, 3:], oshape + (3,))
            fields = np.concatenate((fvecs, areas, fields), axis=-1)
            fields = np.split(fields, strides)
            lr = ZonalLightResult(fields, *axes, pointmetrics=pmetrics)
        return lr

    def _group_query(self, skydata, pm, ptfilter=.25, stol=10, minsun=1):
        pts = [lp.vecs for lp in self.lightplanes if type(lp) != SunsPlaneKD]
        if len(pts) > 0:
            pts = np.vstack(pts)
            pts = pts[pm.in_view(pts, False)]
            pts = pts[translate.cull_vectors(pts, ptfilter)]
            skarea = calc_omega(pts, pm)
        else:
            pts = pm.dxyz.reshape(1, 3)
            skarea = np.array([pm.area])
        # only look for suns when relevant
        sunmask = skydata.sun[:, 3] > 0
        suns = skydata.sun
        # fall back to broadcasting when points will always be the same anyway
        if len(self._issunplane) == 0 or np.sum(sunmask) == 0:
            tidxs, skydatas, dsns, vecs = super()._group_query(skydata, pts)
            vecs = vecs.reshape(len(skydata.maskindices), -1, 6)
            areas = np.broadcast_to(skarea, vecs.shape[0:2]).ravel()
            if len(self._issunplane) > 0:
                tidxs[self._issunplane[0]] = -1
            d = np.zeros(tidxs.shape[1])
            return tidxs, skydatas, dsns, vecs, d, areas

        sunplane = self.lightplanes[self._issunplane[0]]
        # for each sun, matching vectors, indices, and solar errors
        vecs, idx, ds = sunplane.query_by_suns(suns[sunmask, 0:3],
                                               fixed_points=pts,
                                               ptfilter=ptfilter, stol=stol,
                                               minsun=minsun)
        spts = [v[:, 3:] for v in vecs]
        areas = pool_call(calc_omega, spts, pm, expandarg=False,
                          desc="calculating areas")

        sundata = (vecs, idx, ds, areas)
        proxies = ("vecs", np.full(len(pts), -1), np.zeros(len(pts)), skarea)
        smtx, dsns, data2 = self._unmask_data(skydata, sunmask, suns,
                                              pts, sundata, proxies)
        all_vecs, sunidx, d, areas = (np.concatenate(d, axis=0) for d in data2)

        tidxs, skydatas = self._match_ragged(smtx, dsns, sunidx, all_vecs)
        return tidxs, skydatas, dsns, data2[0], d, areas

    def _match_ragged(self, smtx, dsns, sunidx, all_vecs):
        tidxs = []
        skydatas = []
        for i, lp in enumerate(self.lightplanes):
            if i == self._issunplane[0]:
                tidxs.append(sunidx)
                skydatas.append(dsns[:, 3:4])
            else:
                tidxs.append(lp.query(all_vecs[:, 3:])[0])
                skydatas.append(smtx)
        tidxs = np.stack(tidxs)
        return tidxs, skydatas

    def _sinfo(self, serr, vecs, idxs, oshape):
        """error and bin information for evaluate queries"""
        if len(self._issunplane) > 0:
            dinfo = ["src_err", "src_idx"]
            lp = self.lightplanes[self._issunplane[0]]
            idx = idxs[self._issunplane[0]]
            srcidx = np.full(len(idx), -1)
            mask = idx >= 0
            srcidx[mask] = lp.data.idx[idx[mask], 0]
            sinfo = np.broadcast_to(np.stack((serr, srcidx)).T[:, None],
                                    oshape)
        else:
            dinfo = []
            sinfo = None
        return sinfo, dinfo

    @staticmethod
    def _unmask_data(skydata, sunmask, suns, pts, sundata, proxies):
        data2 = [[] for i in range(len(sundata))]
        smtx = []
        dsns = []
        j = 0
        for i, sm in enumerate(sunmask):
            sun = suns[i]
            sky = skydata.smtx[i]
            if sm:
                cnt = len(sundata[0][j])
                for d, d2 in zip(sundata, data2):
                    d2.append(d[j])
                j += 1
            else:
                cnt = len(pts)
                for p, d2 in zip(proxies, data2):
                    if type(p) == str and p == "vecs":
                        s = np.hstack(np.broadcast_arrays(sun[None, 0:3], pts))
                        d2.append(s)
                    else:
                        d2.append(p)
            smtx.append(np.broadcast_to(sky[None, :], (cnt, sky.shape[0])))
            dsns.append(np.broadcast_to(sun[None, :], (cnt, sun.shape[0])))
        smtx = np.vstack(smtx)
        dsns = np.vstack(dsns)
        return smtx, dsns, data2
