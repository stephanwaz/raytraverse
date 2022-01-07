# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import shutil
import sys
from concurrent.futures import wait, FIRST_COMPLETED


import numpy as np
from scipy.spatial import cKDTree, distance_matrix

from raytraverse import translate
from raytraverse.evaluate import MetricSet
from raytraverse.lightfield.lightplanekd import LightPlaneKD
from raytraverse.lightfield.sets import MultiLightPointSet
from raytraverse.lightfield.lightfield import LightField
from raytraverse.lightfield.lightresult import ResultAxis, LightResult
import raytraverse.lightfield._helpers as intg
from raytraverse.mapper import ViewMapper
from raytraverse.utility import pool_call


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

    def __init__(self, scene, vecs, pm, src, includesky=True):
        super().__init__(scene, vecs, pm, src)
        if includesky:
            pts = f"{self._datadir}/sky_points.tsv"
            self._skyplane = LightPlaneKD(self.scene, pts, self.pm, "sky")
        else:
            self._skyplane = None

    @property
    def vecs(self):
        """indexing vectors (sx, sy, sz, px, py, pz)"""
        return self._vecs

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
            shape (N, 6) vectors to query.

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

    def add_indirect_to_suns(self, skyplanedirect, srcprefix="i_"):
        """

        Parameters
        ----------
        skyplanedirect: DayLightPlaneKD
            a skyplane with matching ray samples to self.skyplane
            (use SamplerArea.repeat) representing the direct contribution from
            sky patches.
        srcprefix: str, optional
            if inplace is False, the prefix to add to the source names
        Returns
        -------
        DayLightPlaneKD
            self if inplace is True, else a new DaylightPlaneKD
        """
        if self.skyplane is None:
            raise ValueError("method requires DaylightPlaneKD initialized with"
                             " includesky=True")
        side = int(np.sqrt(self.skyplane.data[0].lum.shape[1] - 1))
        omegar = np.square(0.2665*np.pi*side/180)*.5
        skpatches = translate.xyz2skybin(self.vecs[:, 0:3], side)
        sis, _ = self.skyplane.query(self.vecs[:, 3:])
        args = []
        for i, si in enumerate(sis):
            pf2 = (f"{self.scene.outdir}/{self.skyplane.data[si].parent}/"
                   f"{srcprefix}{self.data[i].src}_points.tsv")
            if not os.path.isfile(pf2):
                args.append((self.data[i], self.skyplane.data[si],
                             skyplanedirect.data[si], skpatches[i]))
        file_depend = pool_call(intg.indirect_to_suns, args, omegar,
                                self.scene, srcprefix=srcprefix,
                                desc="adding indirect to LightPoints")
        file_depend = set(file_depend)
        if None in file_depend:
            file_depend.remove(None)
        for fd in set(file_depend):
            shutil.copy(*fd)
        vecs = np.hstack((self.data.idx[:, 0:1], self.samplelevel[:, 0:1],
                          self.vecs[:, 0:3]))
        src = f"{srcprefix}{self.src}"
        return DayLightPlaneKD(self.scene, vecs, self.pm, src)

    def make_images(self, skydata, points, vm, viewangle=180., res=512,
                    interp=False, prefix="img", namebyindex=False):
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
        vecs, sidx, skp_idx = self._group_query(skydata, points)

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
        outfs = self._process_mgr(skp_idx, sidx, skydata, oshape, True,
                                  intg.img_pt, message="Making Images",
                                  mask_kwargs=mask_kwargs, vms=vms, res=res,
                                  interp=interp, prefix=prefix)
        return sorted([i for j in outfs for i in j])

    def evaluate(self, skydata, points, vm, viewangle=180.,
                 metricclass=MetricSet, metrics=None, datainfo=False,
                 srconly=False, **kwargs):
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
        points = np.atleast_2d(points)
        if hasattr(vm, "dxyz"):
            vms = [vm]
        else:
            dxyz = np.asarray(vm).reshape(-1, 3)
            vm = ViewMapper()
            vms = [ViewMapper(d, viewangle) for d in dxyz]
        if metrics is None:
            metrics = metricclass.defaultmetrics
        oshape = (len(skydata.maskindices), len(points), len(vms), len(metrics))
        vecs, sidx, skp_idx = self._group_query(skydata, points)

        self.scene.log(self, f"Evaluating {oshape[3]} metrics for {oshape[2]}"
                             f" view directions at {oshape[1]} points under "
                             f"{oshape[0]} skies", True)
        sinfo, dinfo = self._sinfo(datainfo, vecs, sidx, points, skp_idx,
                                   oshape)
        # compose axes
        skyaxis = ResultAxis(skydata.maskindices, f"sky")
        ptaxis = ResultAxis(points, "point")
        viewaxis = ResultAxis([v.dxyz for v in vms], "view")
        metricaxis = ResultAxis(list(metrics) + dinfo, "metric")

        # use parallel processing
        sumsafe = metricclass.check_safe2sum(metrics)
        srconly = srconly or skp_idx is None
        fields = self._process_mgr(skp_idx, sidx, skydata, oshape, True,
                                   intg.evaluate_pt,
                                   message="Evaluating Points",
                                   srconly=srconly, sumsafe=sumsafe,
                                   metricclass=metricclass, metrics=metrics,
                                   vm=vm, vms=vms, **kwargs)
        if sinfo is not None:
            nshape = list(sinfo.shape)
            nshape[2] = fields.shape[2]
            sinfo = np.broadcast_to(sinfo, nshape)
            fields = np.concatenate((fields, sinfo), axis=-1)

        lr = LightResult(fields, skyaxis, ptaxis, viewaxis, metricaxis)
        return lr

    def _group_query(self, skydata, points):
        # query and group sun positions
        suns = skydata.sun[:, None, 0:3]
        pts = np.atleast_2d(points)[None, :]
        vecs = np.concatenate(np.broadcast_arrays(suns, pts), -1).reshape(-1, 6)
        # order: (suns, pts) row-major
        sidx, sun_err = self.query(vecs)
        # query sky simulation data
        if self.skyplane is not None:
            skp_idx, sk_pt_err = self.skyplane.query(points)
        else:
            skp_idx = None
        return vecs, sidx, skp_idx

    def _sinfo(self, datainfo, vecs, sidx, points, skp_idx, oshape):
        """error and bin information for evaluate queries"""
        dinfo = ["sun_err", "sun_pt_err", "sun_pt_idx", "sky_pt_err",
                 "sky_pt_idx"]
        if not datainfo:
            dinfo = []
        elif hasattr(datainfo, "__len__"):
            dinfo = [i for i in dinfo if i in datainfo]
        if len(dinfo) == 0:
            return None, dinfo
        if skp_idx is None:
            try:
                dinfo.remove("sky_pt_err")
            except ValueError:
                pass
            try:
                dinfo.remove("sky_pt_idx")
            except ValueError:
                pass
        sinfo = []
        if "sun_err" in dinfo:
            snerr = np.linalg.norm(vecs[:, :3] - self.vecs[sidx, :3],
                                   axis=-1)
            sinfo.append(translate.chord2theta(snerr)*(180/np.pi))
        if "sun_pt_err" in dinfo:
            pterr = np.linalg.norm(vecs[:, 3:] - self.vecs[sidx, 3:],
                                   axis=-1)
            sinfo.append(pterr)
        if "sun_pt_idx" in dinfo:
            sinfo.append(sidx)
        if "sky_pt_err" in dinfo:
            skerr = np.linalg.norm(points - self.skyplane.vecs[skp_idx],
                                   axis=-1)
            sinfo.append(np.tile(skerr, oshape[0]))
        if "sky_pt_idx" in dinfo:
            sinfo.append(np.tile(skp_idx, oshape[0]))
        return np.array(sinfo).T.reshape(*oshape[:-2], 1, -1), dinfo

    def _process_mgr(self, skp_idx, sidx, skydata, oshape, workers, eval_fn,
                     message="Evaluating Points", mask_kwargs=None,
                     **eval_kwargs):
        if skp_idx is None:
            ski = np.full(oshape[0:2], -1).reshape(-1, 1)
        else:
            ski = np.broadcast_to(skp_idx, oshape[0:2]).reshape(-1, 1)
        # tuples of all combinations: order: (suns, pts) row-major
        idx_tup = np.hstack((self.data.idx[sidx], ski))
        # unique returns sorted values (evaluation order)
        qtup, qidx = np.unique(idx_tup, axis=0, return_index=True)
        # broadcast skydata to match full indexing
        s = (*oshape[0:2], skydata.smtx.shape[1])
        smtx = np.broadcast_to(skydata.smtx[:, None, :], s).reshape(-1, s[-1])
        s = (*oshape[0:2], skydata.sun.shape[1])
        dsns = np.broadcast_to(skydata.sun[:, None, :], s).reshape(-1, s[-1])
        s = (*oshape[0:2], 1)
        dprx = np.broadcast_to(skydata.sunproxy[:, None, None], s).reshape(-1, s[-1])
        tup_sort = np.lexsort(idx_tup.T[::-1])
        # make an inverse sort to undo evaluation order
        tup_isort = np.argsort(tup_sort, kind='stable')

        fields = []
        with self.scene.progress_bar(self, message=message,
                                     total=len(qtup), workers=workers) as pbar:
            exc = pbar.pool
            pbar.write(f"Calculating {len(qidx)} sun/sky/pt combinations",
                       file=sys.stderr)
            futures = []
            done = set()
            not_done = set()
            cnt = 0
            pbar_t = 0
            # submit asynchronous to process pool
            for qi, skp_qi in zip(sidx[qidx], qtup):
                if skp_idx is None:
                    skp = None
                else:
                    skp = self.skyplane.data[skp_qi[2]]
                snp = self.data[qi]
                mask = np.all(idx_tup == skp_qi, -1)
                # manage to queue to avoid loading too many points in memory
                # and update progress bar as completed
                if cnt > pbar.nworkers*3:
                    wait_r = wait(not_done, return_when=FIRST_COMPLETED)
                    not_done = wait_r.not_done
                    done.update(wait_r.done)
                    pbar.update(len(done) - pbar_t)
                    pbar_t = len(done)
                if mask_kwargs is not None:
                    for k, v in mask_kwargs.items():
                        eval_kwargs.update([(k, v[mask])])
                fu = exc.submit(eval_fn, skp, snp, smtx[mask], dsns[mask],
                                dprx[mask], **eval_kwargs)
                futures.append(fu)
                not_done.add(fu)
                cnt += 1
            # gather results (in order)
            for future in futures:
                fields.append(future.result())
                if future in not_done:
                    pbar.update(1)
            pbar.write(f"Completed evaluation for {len(idx_tup)} values",
                       file=sys.stderr)
            # sort back to original order and reshape
            if type(fields[0]) != list:
                fields = np.concatenate(fields, axis=0)[tup_isort].reshape(oshape)
        return fields
