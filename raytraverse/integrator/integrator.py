# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import sys
from concurrent.futures import wait, FIRST_COMPLETED


import numpy as np

from raytraverse import translate
from raytraverse.evaluate import MetricSet
from raytraverse.lightfield.lightresult import ResultAxis, LightResult
import raytraverse.integrator._helpers as intg
from raytraverse.mapper import ViewMapper


class Integrator(object):
    """collection of lightplanes with KDtree structure for sun position query

    Parameters
    ----------
    lightplanes: Sequence[raytraverse.lightfield.LightPlaneKD]
    """

    def __init__(self, *lightplanes, includesky=True, includesun=True):
        self.scene = lightplanes[0].scene
        self.lightplanes = lightplanes
        self.includesky = includesky
        self.includesun = includesun

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
        vecs, idxs = self._group_query(skydata, points)

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
        outfs = self._process_mgr(idxs, skydata, oshape, True,
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
            values include: ["pt_err", "pt_idx", "src_err", "src_idx"].
            If True, includes all. "err" is distance from queried vector to
            actual. "bin" is the unraveled idx of source vector at a 500^2
            resolution of the mapper.

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
        vecs, idxs = self._group_query(skydata, points)

        self.scene.log(self, f"Evaluating {oshape[3]} metrics for {oshape[2]}"
                             f" view directions at {oshape[1]} points under "
                             f"{oshape[0]} skies", True)
        sinfo, dinfo = self._sinfo(datainfo, vecs, points, idxs,
                                   oshape)
        # compose axes
        skyaxis = ResultAxis(skydata.maskindices, f"sky")
        ptaxis = ResultAxis(points, "point")
        viewaxis = ResultAxis([v.dxyz for v in vms], "view")
        metricaxis = ResultAxis(list(metrics) + dinfo, "metric")

        # use parallel processing
        sumsafe = metricclass.check_safe2sum(metrics)
        fields = self._process_mgr(idxs, skydata, oshape, True,
                                   intg.evaluate_pt,
                                   message="Evaluating Points", srconly=srconly,
                                   sumsafe=sumsafe, metricclass=metricclass,
                                   metrics=metrics, vm=vm, vms=vms, **kwargs)
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
        idxs = []
        for lp in self.lightplanes:
            if lp.vecs.shape[1] == 6:
                idx, err = lp.query(vecs)
            else:
                idx, err = lp.query(points)
            idxs.append(idx)
        return vecs, idxs

    def _sinfo(self, datainfo, vecs, points, idxs, oshape):
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
                elif di == "pt_err" and lp.vecs.shape[1] == 6:
                    pterr = np.linalg.norm(vecs[:, 3:] - lp.vecs[idx, 3:],
                                           axis=-1)
                    sinfo.append(pterr)
                    dinfo2.append(f"{lp.src}_{di}")
                elif di == "pt_err":
                    pterr = np.linalg.norm(points - lp.vecs[idx],
                                           axis=-1)
                    sinfo.append(np.tile(pterr, oshape[0]))
                    dinfo2.append(f"{lp.src}_{di}")
                elif di == "pt_idx" and lp.vecs.shape[1] == 6:
                    ptidx = lp.data.idx[idx, 1]
                    sinfo.append(ptidx)
                    dinfo2.append(f"{lp.src}_{di}")
                elif di == "pt_idx":
                    sinfo.append(np.tile(idx, oshape[0]))
                    dinfo2.append(f"{lp.src}_{di}")
        return np.array(sinfo).T.reshape(*oshape[:-2], 1, -1), dinfo2

    def _build_run_data(self, idxs, skydata, oshape):
        # broadcast skydata to match full indexing
        s = (*oshape[0:2], skydata.smtx.shape[1])
        smtx = np.broadcast_to(skydata.smtx[:, None, :], s).reshape(-1, s[-1])
        ds = (*oshape[0:2], skydata.sun.shape[1])
        dsns = np.broadcast_to(skydata.sun[:, None, :], ds).reshape(-1, ds[-1])
        # makes coefficient list and fill idx lists
        tidxs = []
        skydatas = []

        # generic, assumes sun and/or sky, uses patch sun when no sun present
        skyi = None
        suni = None
        for i, (lp, idx) in enumerate(zip(self.lightplanes, idxs)):
            if lp.vecs.shape[1] == 3:
                tidxs.append(np.broadcast_to(idx, oshape[0:2]).ravel())
                skyi = i
                skydatas.append(smtx)
            elif self.includesun:
                tidxs.append(idx)
                suni = i
                skydatas.append(dsns[:, 3:4])
        if self.includesun and suni is None and skyi is not None:
            dprx = skydata.smtx_patch_sun(includesky=self.includesky)
            dprx = np.broadcast_to(dprx[:, None, :], s).reshape(-1, s[-1])
            skydatas[skyi] = dprx
        return tidxs, skydatas, dsns

    def _process_mgr(self, idxs, skydata, oshape, workers, eval_fn,
                     message="Evaluating Points", mask_kwargs=None,
                     **eval_kwargs):

        tidxs, skydatas, dsns = self._build_run_data(idxs, skydata, oshape)

        # tuples of all combinations: order: (suns, pts) row-major
        idx_tup = np.stack(tidxs).T
        # unique returns sorted values (evaluation order)
        qtup, qidx = np.unique(idx_tup, axis=0, return_index=True)
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
            for qi, qt in zip(qidx, qtup):
                lpts = []
                for lp, tidx in zip(self.lightplanes, tidxs):
                    lpts.append(lp.data[tidx[qi]])
                mask = np.all(idx_tup == qt, -1)
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
                sx = [smx[mask] for smx in skydatas]
                fu = exc.submit(eval_fn, lpts, sx, dsns[mask], **eval_kwargs)
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
