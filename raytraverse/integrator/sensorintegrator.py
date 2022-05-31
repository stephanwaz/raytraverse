# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse.lightfield import ZonalLightResult, ResultAxis, LightResult
from raytraverse.integrator.integrator import Integrator


class SensorIntegrator(Integrator):
    """collection of sensorplanes with evaluation routines

    Parameters
    ----------
    lightplanes: Sequence[raytraverse.lightfield.SensorPlaneKD]
    ptype: Sequence[str]
        raises
        matching order of lightplanes, requires one for each:
            - "sky": represents sky with nsrcs = skydata
            - "skysun": represents sky+sun with nsrcs = skydata
            - "patch": represents sun contribution as patch
            - "sun": sun contribution
            - "fixed": does not respond to skydata (electric lighting)
    factors: Sequence[int], optional
        values, for each light plane to scale contribution of eeach light plane
        for example, provide (1, -1, 1) for ptype: ("skysun", "patch", "sun")
        for 2-phase DDS calculation. If not give, all are set to 1
    """

    def __init__(self, *lightplanes, ptype=None, factors=None, **kwargs):
        # argument checking
        if ptype is None:
            raise ValueError("ptype is required to initialize SensorIntegrator")
        if len(ptype) != len(lightplanes):
            raise ValueError(f"length of ptype ({len(ptype)}) does not match "
                             f"# of lightplanes ({len(lightplanes)}")
        ptypes = ("sky", "skysun", "patch", "sun", "fixed")
        badtypes = [i for i in ptype if i not in ptypes]
        if len(badtypes) > 0:
            raise ValueError(f"Invalid value(s) for ptype: {badtypes} not in "
                             f"{ptypes}")
        if factors is None:
            factors = [1] * len(lightplanes)
        if len(factors) != len(lightplanes):
            raise ValueError(f"length of factors ({len(factors)}) does not "
                             f"match # of lightplanes ({len(lightplanes)}")
        # make sure no sunviewengine is passed
        kwargs.update(sunviewengine=None)
        # set sensor specific args
        self.sensors = lightplanes[0].sensors
        self.factors = factors
        self.ptype = ptype
        super().__init__(*lightplanes, **kwargs)

    def make_images(self, *args, **kwargs):
        raise ValueError("SensorIntegrator does not make_images")

    def evaluate(self, skydata, points=None, vm=None, datainfo=False,
                 ptfilter=.25, stol=10,  minsun=1, **kwargs):
        """apply sky data and view queries to daylightplane to return metrics
        parallelizes and optimizes run order.

        Parameters
        ----------
        skydata: raytraverse.sky.Skydata
        points: np.array, optional
            shape (N, 3), if None evaluates zone
        vm: ignored
        datainfo: Union[Sized[str], bool], optional
            include information about source data as additional metrics. Valid
            values include: ["pt_err", "pt_idx", "src_err", "src_idx"].
            If True, includes all. zonal evaluation will only include src_err
            and src_idx
        ptfilter: Union[float, int], optional
            minimum seperation for returned points
        stol: Union[float, int], optional
            maximum angle (in degrees) for matching sun vectors
        minsun: int, optional
            if atleast these many suns are not returned based on stol, directly
            query for this number of results (regardless of sun error)

        Returns
        -------
        raytraverse.lightfield.LightResult
        """
        # delegate to zonal_evaluate
        if (points is None and len(self._issunplane) > 0 and
                np.any(skydata.sun[:, 3] > 0)):
            return self.zonal_evaluate(skydata, self.lightplanes[0].pm,
                                       datainfo=datainfo, ptfilter=ptfilter,
                                       stol=stol, minsun=minsun, **kwargs)
        if points is None:
            points, skarea = self._get_fixed_points(self.lightplanes[0].pm,
                                                    ptfilter)
            # include areas with points when available
            apoints = np.hstack((points, skarea[:, None]))
        else:
            points = np.atleast_2d(points)
            apoints = points

        sensors = self.sensors

        tidxs, skydatas, vecs = self._group_query(skydata, points)
        oshape = (len(skydata.maskindices), len(points), len(sensors), 1)
        self.scene.log(self, f"Evaluating {oshape[2]} sensors at {oshape[1]} "
                             f"points under {oshape[0]} skies", True)

        fields = None
        for lp, sd, ti in zip(self.lightplanes, skydatas, tidxs):
            print(ti.shape, sd.shape, lp.data[ti].shape)
            if len(ti) == len(points):
                data = np.einsum("hs,pnsf->hpnf", sd, lp.data[ti])
            else:
                d = lp.data[ti].reshape(*oshape[0:3], sd.shape[1], oshape[3])
                data = np.einsum("hs,hpnsf->hpnf", sd, d)
            if fields is None:
                fields = data
            else:
                fields += data
        fields = fields.reshape(oshape)

        sinfo, dinfo = self._sinfo(datainfo, vecs, tidxs, oshape[0:2])
        if sinfo is not None:
            nshape = list(sinfo.shape)
            nshape[2] = fields.shape[2]
            sinfo = np.broadcast_to(sinfo, nshape)
            fields = np.concatenate((fields, sinfo), axis=-1)
        # compose axes: (skyaxis, ptaxis, viewaxis, metricaxis)
        axes = (ResultAxis(skydata.rowlabel[skydata.fullmask], f"sky"),
                ResultAxis(apoints, "point"),
                ResultAxis(sensors, "view"),
                ResultAxis(["illum"] + dinfo, "metric"))
        lr = LightResult(fields, *axes)
        return lr

    def zonal_evaluate(self, skydata, pm, vm=None, datainfo=False,
                       ptfilter=.25, stol=10, minsun=1, **kwargs):
        """
        Parameters
        ----------
        see evaluate

        Returns
        -------
        raytraverse.lightfield.ZonalLightResult
        """
        # delegate back to evaluate
        if len(self._issunplane) == 0:
            return self.evaluate(skydata, datainfo=datainfo, ptfilter=ptfilter,
                                 stol=stol, minsun=minsun, **kwargs)

        self.scene.log(self, f"Evaluating {len(self.sensors)} sensors across "
                             f"zone {pm.name} under {len(skydata.maskindices)} "
                             f"skies", True)
        (tidxs, skydatas, dsns, vecs, serr,
         areas, pts, cnts) = self._zonal_group_query(skydata, pm,
                                                     ptfilter=ptfilter,
                                                     stol=stol, minsun=minsun)
        chunks = round(tidxs[0].size/525000)
        fields = np.zeros((len(tidxs[0]), len(self.sensors), 1))
        for lp, sd, ti in zip(self.lightplanes, skydatas, tidxs):
            if chunks > 1:
                args = zip(np.array_split(sd, chunks),
                           np.array_split(ti, chunks))
                data = []
                for sd_c, ti_c in args:
                    data.append(np.einsum("ps,pnsf->pnf", sd_c, lp.data[ti_c]))
                fields += np.concatenate(data, axis=0)
            else:
                fields += np.einsum("ps,pnsf->pnf", sd, lp.data[ti])

        oshape = (len(fields), len(self.sensors))
        pmetrics = ['x', 'y', 'z', 'area']
        if datainfo:
            sinfo, dinfo = self._zonal_sinfo(serr, tidxs, oshape + (2,))
            if sinfo is not None:
                fields = np.concatenate((sinfo, fields), axis=-1)
                pmetrics += dinfo

        areas = np.broadcast_to(areas[:, None, None], oshape + (1,))
        axes = [ResultAxis(skydata.rowlabel[skydata.fullmask], f"sky"),
                ResultAxis([pm.name], f"zone"),
                ResultAxis(self.sensors, "view"),
                ResultAxis(pmetrics + ["illum"], "metric")]

        strides = np.cumsum(cnts)[:-1]
        fvecs = np.broadcast_to(vecs[:, None, 3:], oshape + (3,))
        fields = np.concatenate((fvecs, areas, fields), axis=-1)
        fields = np.split(fields, strides)
        return ZonalLightResult(fields, *axes, pointmetrics=pmetrics)

    def _group_query(self, skydata, points):
        """for sensor integration broadcasting happens in the summation, so
        this function is just allocating the correct points and sky matrix
        only vecs is broadcast to the actual result size"""
        # query and group sun positions
        suns = skydata.sun[:, None, 0:3]
        pts = np.atleast_2d(points)[None, :]
        vecs = np.concatenate(np.broadcast_arrays(suns, pts), -1).reshape(-1, 6)
        skydatas = self._select_sky_grid(skydata)
        idxs = []
        for lp in self.lightplanes:
            if lp.vecs.shape[1] == 6:
                idxs.append(lp.query(vecs)[0])
            else:
                idxs.append(lp.query(points)[0])
        return idxs, skydatas, vecs

    def _select_sky_grid(self, skydata):
        skydatas = []
        for p, f, lp in zip(self.ptype, self.factors, self.lightplanes):
            # broadcast skydata to match full indexing
            if p == "sky":
                skydatas.append(skydata.smtx * f)
            elif p == "skysun":
                skydatas.append(skydata.smtx_patch_sun() * f)
            elif p == "patch":
                skydatas.append(skydata.smtx_patch_sun(False) * f)
            elif p == "sun":
                skydatas.append(skydata.sun[:, 3:4] * f)
            else:
                skydatas.append(np.full((len(skydata.sun), 1, 1), f))
        return skydatas

