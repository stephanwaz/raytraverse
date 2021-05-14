# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from concurrent.futures import wait, FIRST_COMPLETED


import numpy as np
from scipy.spatial import cKDTree, distance_matrix

from raytraverse import translate
from raytraverse.evaluate import MetricSet
from raytraverse.lightfield.lightplanekd import LightPlaneKD
from raytraverse.lightfield.sets import MultiLightPointSet
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
    def vecs(self):
        """indexing vectors (such as position, sun positions, etc.)"""
        return self._vecs

    @property
    def samplelevel(self):
        """the level at which the vec was sampled (all zero if not provided
        upon initialization"""
        return self._samplelevel

    @vecs.setter
    def vecs(self, pt):
        pts, idx, samplelevel = self._load_vecs(pt)
        # calculate sun sampling resolution for weighting query vecs
        s0 = pts[samplelevel == 0]
        dm = distance_matrix(s0, s0)
        cm = np.ma.MaskedArray(dm, np.eye(dm.shape[0]))
        sund = np.average(np.min(cm, axis=0).data)
        self._normalization = sund / self.pm.ptres * 2 * 2**.5
        s_pts = []
        s_idx = []
        s_lev = []
        for i, pt, sl in zip(idx, pts, samplelevel):
            source = f"{self.src}_{i:04d}"
            ptf = f"{self.scene.outdir}/{self.pm.name}/{source}_points.tsv"
            spt, sidx, slev = self._load_vecs(ptf)
            s_pts.append(np.hstack((np.broadcast_to(pt[None], spt.shape), spt)))
            s_idx.append(np.stack((np.broadcast_to([i], sidx.shape), sidx)).T)
            s_lev.append(np.stack((np.broadcast_to([sl], slev.shape), slev)).T)
        self._vecs = np.concatenate(s_pts)
        self._kd = None
        self._samplelevel = np.concatenate(s_lev)
        self.omega = None
        self.data = np.concatenate(s_idx)

    @property
    def data(self):
        """LightPlaneSet"""
        return self._data

    @data.setter
    def data(self, idx):
        self._data = MultiLightPointSet(self.scene, self.vecs, idx, self.src,
                                        self.pm.name)

    @property
    def kd(self):
        """kdtree for spatial queries built on demand"""
        if self._kd is None:
            weighted_vecs = np.copy(self.vecs)
            weighted_vecs[:, 3:] *= self._normalization
            self._kd = cKDTree(weighted_vecs)
        return self._kd

    @property
    def skyplane(self):
        """LightPlaneKD of sky results"""
        return self._skyplane

    def query(self, vecs):
        """return the index and distance of the nearest vec to each of vecs

        Parameters
        ----------
        vecs: np.array
            shape (N, 3) vectors to query.

        Returns
        -------
        i: np.array
            integer indices of closest ray to each query
        d: np.array
            distance from query to point, positional distance is normalized
            by the average chord-length between leveel 0 sun samples divided by
            the PlanMapper ptres * sqrt(2).
        """
        weighted_vecs = np.copy(np.atleast_2d(vecs))
        weighted_vecs[:, 3:] *= self._normalization
        d, i = self.kd.query(weighted_vecs)
        return i, d

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
        if hasattr(vm, "dxyz"):
            vms = [vm]
        else:
            dxyz = np.asarray(vm).reshape(-1, 3)
            vm = ViewMapper()
            vms = [ViewMapper(d, 180) for d in dxyz]
        if metrics is None:
            metrics = metricclass.defaultmetrics

        oshape = (len(skydata.maskindices), len(points), len(vms), len(metrics))
        self.scene.log(self, f"Evaluating {oshape[3]} metrics for {oshape[2]}"
                             f" view directions at {oshape[1]} points under "
                             f"{oshape[0]} skies", logerr)

        # query and group sun positions
        suns = skydata.sun[:, None, 0:3]
        pts = np.atleast_2d(points)[None, :]
        vecs = np.concatenate(np.broadcast_arrays(suns, pts), -1).reshape(-1, 6)
        # order: (suns, pts) row-major
        sidx, sun_err = self.query(vecs)
        # query sky simulation data
        skp_idx, sk_pt_err = self.skyplane.query(points)

        # use parallel processing
        fields = self._eval_mgr(skp_idx, sidx, skydata, oshape,
                                vm=vm, vms=vms, metricclass=metricclass,
                                metrics=metrics, **kwargs)
        sinfo, dinfo = self._sinfo(datainfo, vecs, sidx, points, skp_idx,
                                   oshape)
        if sinfo is not None:
            fields = np.concatenate((fields, sinfo), axis=-1)
        # compose axes
        skyaxis = ResultAxis(skydata.maskindices, f"sky")
        ptaxis = ResultAxis(points, "point")
        viewaxis = ResultAxis([v.dxyz for v in vms], "view")
        metricaxis = ResultAxis(list(metrics) + dinfo, "metric")
        lr = LightResult(fields, skyaxis, ptaxis, viewaxis, metricaxis)
        self.scene.log(self, f"Completed evaluation for {fields.size} values",
                       logerr)
        return lr

    def _eval_mgr(self, skp_idx, sidx, skydata, oshape, srconly=False, **kwargs):
        """sort and submit queries to process pool"""
        ski = np.broadcast_to(skp_idx, oshape[0:2]).reshape(-1, 1)
        # tuples of all combinations: order: (suns, pts) row-major
        idx_tup = np.hstack((self.data.idx[sidx], ski))
        tup_sort = np.lexsort(idx_tup.T[::-1])
        # make an inverse sort to undo evaluation order
        tup_isort = np.argsort(tup_sort, kind='stable')

        # unique returns sorted values (evaluation order)
        qtup, qidx = np.unique(idx_tup, axis=0, return_index=True)
        # broadcast skydata to match full indexing
        s = (*oshape[0:2], skydata.smtx.shape[1])
        smtx = np.broadcast_to(skydata.smtx[:, None, :], s).reshape(-1, s[-1])
        s = (*oshape[0:2], skydata.sun.shape[1])
        dsns = np.broadcast_to(skydata.sun[:, None, :], s).reshape(-1, s[-1])

        fields = []
        if srconly:
            workers = "threads"
        else:
            workers = True
        with self.scene.progress_bar(self, "Evaluating Points", len(qtup),
                                     workers=workers) as (exc, pbar):
            pbar.write(f"Calculating {len(qidx)} sun/sky/pt combinations")
            futures = []
            done = set()
            not_done = set()
            cnt = 0
            pbar_t = 0
            # submit asynchronous to process pool
            for qi, skp_qi in zip(sidx[qidx], qtup):
                skp = self.skyplane.data[skp_qi[2]]
                snp = self.data[qi]
                mask = np.all(idx_tup == skp_qi, -1)
                # manage to queue to avoid loading too many points in memory
                # and update progress bar as completed
                if cnt > exc._max_workers*3:
                    wait_r = wait(not_done, return_when=FIRST_COMPLETED)
                    not_done = wait_r.not_done
                    done.update(wait_r.done)
                    pbar.update(len(done) - pbar_t)
                    pbar_t = len(done)
                fu = exc.submit(_evaluate_pt, skp, snp, smtx[mask],
                                dsns[mask], srconly=srconly, **kwargs)
                futures.append(fu)
                not_done.add(fu)
                cnt += 1
            # gather results (in order)
            for future in futures:
                fields.append(future.result())
                if future in not_done:
                    pbar.update(1)
        # sort back to original order and reshape
        fields = np.concatenate(fields, axis=0)[tup_isort].reshape(oshape)
        return fields

    def _sinfo(self, datainfo, vecs, sidx, points, skp_idx, oshape):
        """error and bin information for evaluate queries"""
        dinfo = ["sun_pt_err", "sun_pt_bin", "sky_pt_err", "sky_pt_bin",
                 "sun_err", "sun_bin"]
        if not datainfo:
            dinfo = []
        elif hasattr(datainfo, "__len__"):
            dinfo = [i for i in dinfo if i in datainfo]
        if len(dinfo) == 0:
            return None, dinfo
        sinfo = []
        if "sun_pt_err" in dinfo:
            pterr = np.linalg.norm(vecs[:, 3:] - self.vecs[sidx, 3:],
                                   axis=-1)
            sinfo.append(pterr)
        if "sun_pt_bin" in dinfo:
            ptbin = self.pm.uv2idx(self.pm.xyz2uv(self.vecs[sidx, 3:]),
                                   self.pm.framesize(500))
            sinfo.append(ptbin)
        if "sky_pt_err" in dinfo:
            skerr = np.linalg.norm(points - self.skyplane.vecs[skp_idx],
                                   axis=-1)
            sinfo.append(np.tile(skerr, oshape[0]))
        if "sky_pt_bin" in dinfo:
            ptbin = self.pm.uv2idx(self.pm.xyz2uv(
                self.skyplane.vecs[skp_idx]),
                self.pm.framesize(500))
            sinfo.append(np.tile(ptbin, oshape[0]))
        if "sun_err" in dinfo:
            snerr = np.linalg.norm(vecs[:, :3] - self.vecs[sidx, :3],
                                   axis=-1)
            sinfo.append(translate.chord2theta(snerr)*(180/np.pi))
        if "sun_bin" in dinfo:
            sm = SkyMapper()
            sun_bin = sm.uv2idx(sm.xyz2uv(self.vecs[sidx, :3]), (500, 500))
            sinfo.append(sun_bin)
        return np.array(sinfo).T.reshape(*oshape[:-2], 1, -1), dinfo


def _evaluate_pt(skpoint, snpoint, skyvecs, suns, vm=None, vms=None,
                 metricclass=None, metrics=None, srconly=False, **kwargs):
    """point by point evaluation suitable for submitting to ProcessPool"""
    if srconly:
        sunskypt = snpoint
        smtx = suns[:, 3]
    else:
        sunskypt = skpoint.add(snpoint)
        smtx = np.hstack((skyvecs, suns[:, 3:4]))
    if len(vms) == 1:
        didx = sunskypt.query_ball(vms[0].dxyz,
                                   vms[0].viewangle * vms[0].aspect)[0]
    else:
        didx = None
    pts = []
    for skyvec, sun in zip(smtx, suns):
        vol = sunskypt.evaluate(skyvec, vm=vm, idx=didx,
                                srcvecoverride=sun[0:3], srconly=srconly)
        views = []
        for v in vms:
            views.append(metricclass(*vol, v, metricset=metrics,
                                     **kwargs)())
        views = np.stack(views)
        pts.append(views)
    pts = np.stack(pts)
    return pts
