# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from clasp.script_tools import pool_call

from raytraverse import translate
from raytraverse.evaluate import MetricSet
from raytraverse.lightfield.lightplanekd import LightPlaneKD
from raytraverse.lightfield.sets import LightPlaneSet
from raytraverse.lightfield.lightfield import LightField
from raytraverse.lightfield.lightresult import ResultAxis, LightResult
from raytraverse.mapper import ViewMapper, SkyMapper


class DayLightPlaneKD(LightField):
    """collection of lightplanes with KDtree structure for sun position query

    Parameters
    ----------
    scene: raytraverse.scene.BaseScene
    vecs: np.array str
        suns as array or file shape (N,3), (N,4) or (N,5) if 3, indexed from 0
    pm: raytraverse.mapper.PlanMapper
    src: str
        name of sun sources group.
    """

    def __init__(self, scene, vecs, pm, src):
        super().__init__(scene, vecs, pm, src)
        pts = f"{self._datadir}/sky_points.tsv"
        self._skyplane = LightPlaneKD(self.scene, pts, self.pm, "sky")

    @property
    def data(self):
        """LightPlaneSet"""
        return self._data

    @property
    def skyplane(self):
        return self._skyplane

    @data.setter
    def data(self, idx):
        self._data = LightPlaneSet(LightPlaneKD, self.scene, self.pm, idx,
                                   self.src)

    def query(self, vecs):
        """return the index and distance of the nearest point to each of points

        Parameters
        ----------
        vecs: np.array
            shape (N, 3) vectors to query.

        Returns
        -------
        i: np.array
            integer indices of closest ray to each query
        d: np.array
            distance from query to point in spacemapper.
        """
        d, i = self.kd.query(vecs)
        d = translate.chord2theta(d) * 180/np.pi
        return i, d

    def query_ball(self, vecs, viewangle=10):
        """return set of rays within a view cone

        Parameters
        ----------
        vecs: np.array
            shape (N, 3) vectors to query.
        viewangle: int float
            opening angle of view cone

        Returns
        -------
        i: list np.array
            if vecs is a single vector, a list of indices within radius.
            if vecs is a set of points an array of lists, one
            for each is returned.
        """
        vs = translate.theta2chord(viewangle/360*np.pi)
        return self.kd.query_ball_point(translate.norm(vecs), vs)

    def evaluate(self, skydata, points, vm, metricclass=MetricSet,
                 metrics=None, logerr=True, datainfo=False, **kwargs):
        """

        Parameters
        ----------
        skydata: raytraverse.sky.Skydata
        points: np.array
        vm: Union[raytraverse.mapper.ViewMapper, np.array]
        metricclass: raytraverse.evaluate.BaseMetricSet, optional
        metrics: Sized, optional
        logerr: bool, optional
        datainfo: Union[Sized[str], bool], optional
            include information about source data as additional metrics. Valid
            values include: ["sun_pt_err", "sun_pt_bin", "sky_pt_err",
            "sky_pt_bin", "sun_err", "sun_bin"]. If True, includes all. "err" is
            distance from queried vector to actual. "bin" is the unraveled idx
            of source vector at a 500^2 resolution of the mapper.
            order is ignored, info is always in order listed above after the
            last metric.


        Returns
        -------
        raytraverse.lightfield.LightResult
        """
        dinfo = ["sun_pt_err", "sun_pt_bin", "sky_pt_err", "sky_pt_bin",
                 "sun_err", "sun_bin"]
        if not datainfo:
            dinfo = []
        elif hasattr(datainfo, "__len__"):
            dinfo = [i for i in dinfo if i in datainfo]
        if hasattr(vm, "dxyz"):
            vms = [vm]
        else:
            dxyz = np.asarray(vm).reshape(-1, 3)
            vm = ViewMapper()
            vms = [ViewMapper(d, 180) for d in dxyz]

        # compose axes
        skyaxis = ResultAxis(skydata.maskindices, f"sky")
        ptaxis = ResultAxis(points, "point")
        viewaxis = ResultAxis([v.dxyz for v in vms], "view")
        if metrics is None:
            metrics = metricclass.defaultmetrics
        metricaxis = ResultAxis(list(metrics) + dinfo, "metric")
        axes = (skyaxis, ptaxis, viewaxis, metricaxis)

        self.scene.log(self, f"Evaluating {len(metrics)} metrics for {len(vms)}"
                             f" view directions at {len(points)} points under "
                             f"{len(skydata.maskindices)} skies", logerr)
        # query and group sun positions
        np.set_printoptions(linewidth=200)
        sidx, sun_err = self.query(skydata.sun[:, 0:3])
        # get sort order to match evaluation order and inverse
        sun_sort = np.argsort(sidx, kind='stable')
        sun_isort = np.argsort(sun_sort, kind='stable')
        # unique returns sorted values
        dp_qidx = np.unique(sidx)
        # query sky simulation data
        skp_idx, sk_pt_err = self.skyplane.query(points)
        # decide whether to split processors by sun positions or points
        pool_suns = len(dp_qidx) > len(points)
        plkwargs = dict(skp=self.skyplane.data, skp_idx=skp_idx, points=points,
                        vm=vm, vms=vms, sp=pool_suns, dinfo=dinfo,
                        metricclass=metricclass, metrics=metrics, **kwargs)
        args = []
        # iterate through suns
        for qi in dp_qidx:
            mask = sidx == qi
            args.append((self.data[qi], skydata.smtx[mask], skydata.sun[mask]))
        fields = pool_call(_evaluate_pls, args, kwargs=plkwargs, expand=True,
                           test=not pool_suns)
        fields = np.concatenate(fields)[sun_isort]

        # add globally consistent info
        sinfo = []
        if "sky_pt_err" in dinfo:
            sinfo.append(np.broadcast_to(sk_pt_err[None, :, None, None],
                                         fields.shape[:-1] + (1,)))
        if "sky_pt_bin" in dinfo:
            ptbin = self.pm.uv2idx(self.pm.xyz2uv(self.skyplane.vecs[skp_idx]),
                                   self.pm.framesize(500))
            sinfo.append(np.broadcast_to(ptbin[None, :, None, None],
                                         fields.shape[:-1] + (1,)))
        if "sun_err" in dinfo:
            sinfo.append(np.broadcast_to(sun_err[:, None, None, None],
                                         fields.shape[:-1] + (1,)))
        if "sun_bin" in dinfo:
            sm = SkyMapper(sunres=1)
            sun_bin = sm.uv2idx(sm.xyz2uv(self.vecs[sidx]), (500, 500))
            sinfo.append(np.broadcast_to(sun_bin[:, None, None, None],
                                         fields.shape[:-1] + (1,)))
        if len(sinfo) > 0:
            fields = np.concatenate((fields, *sinfo), axis=-1)


        # inverse sort fields to match input order
        lr = LightResult(fields, *axes)
        ev = int(lr.data.size * (len(metrics) / lr.data.shape[-1]))
        self.scene.log(self, f"Completed Evaluation for {ev} items",
                       logerr)
        return lr


def _evaluate_pls(snplane, skyvecs, suns, skp=None, skp_idx=None, points=None,
                  sp=False, dinfo=None, **kwargs):
    """plane by plane evaluation suitable for submitting to ProcessPool

    Parameters
    ----------
    snplane
    skyvecs
    suns
    skp
    skp_idx
    points
    sp
    dinfo
    kwargs

    Returns
    -------

    """
    # query sun simulation data
    snp_idx, sn_pt_err = snplane.query(points)
    # find unique combinations
    idx_pair = np.stack((skp_idx, snp_idx))
    # unique returns sorted values
    sn_qidx, sn_midx = np.unique(idx_pair.T, axis=0, return_inverse=True)
    pkwargs = dict(skyvecs=skyvecs, suns=suns, **kwargs)
    # iterate through points
    args = [(skp[sk_qi], snplane.data[sn_qi]) for sk_qi, sn_qi
            in sn_qidx]
    planes = pool_call(_evaluate_pts, args, kwargs=pkwargs, expand=True,
                       test=sp)
    planes = np.stack(planes, axis=1)
    planes = planes[:, sn_midx]

    # add sunplane unique info
    sinfo = []
    if "sun_pt_err" in dinfo:
        sinfo.append(np.broadcast_to(sn_pt_err[None, :, None, None],
                                     planes.shape[:-1] + (1,)))
    if "sun_pt_bin" in dinfo:
        ptbin = snplane.pm.uv2idx(snplane.pm.xyz2uv(snplane.vecs[snp_idx]),
                                  snplane.pm.framesize(500))
        sinfo.append(np.broadcast_to(ptbin[None, :, None, None],
                                     planes.shape[:-1] + (1,)))
    if len(sinfo) > 0:
        planes = np.concatenate((planes, *sinfo), axis=-1)
    return planes


def _evaluate_pts(skpoint, snpoint, vm=None, vms=None, skyvecs=None, suns=None,
                  metricclass=None, metrics=None, **kwargs):
    """point by point evaluation suitable for submitting to ProcessPool

    Parameters
    ----------
    skpoint
    snpoint
    vm
    vms
    skyvecs
    suns
    metricclass
    metrics
    kwargs

    Returns
    -------

    """
    sunskypt = skpoint.add(snpoint)
    if len(vms) == 1:
        didx = sunskypt.query_ball(vms[0].dxyz,
                                   vms[0].viewangle * vms[0].aspect)[0]
    else:
        didx = None
    pts = []
    for skyvec, sun in zip(skyvecs, suns):
        avec = np.concatenate((skyvec, sun[3:4]))
        vol = sunskypt.evaluate(avec, vm=vm, idx=didx,
                                srcvecoverride=sun[0:3])
        views = []
        for v in vms:
            views.append(metricclass(*vol, v, metricset=metrics,
                                     **kwargs)())
        views = np.stack(views)
        pts.append(views)
    return np.stack(pts)
