# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import translate
from raytraverse.lightfield.lightplanekd import LightPlaneKD
from raytraverse.lightfield.sets import LightPlaneSet
from raytraverse.lightfield.lightfield import LightField


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
        self._skydata = LightPlaneKD(self.scene, pts, self.pm, "sky")

    @property
    def data(self):
        """LightPlaneSet"""
        return self._data

    @property
    def skydata(self):
        return self._skydata

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


