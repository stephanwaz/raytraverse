# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================


class LightField(object):
    """container for accessing sampled data

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    rebuild: bool, optional
        build kd-tree even if one exists
    prefix: str, optional
        prefix of data files to map
    """

    def __init__(self, scene, rebuild=False, prefix='light'):
        #: bool: force rebuild kd-tree
        self.rebuild = rebuild
        #: str: prefix of data files from sampler (stype)
        self.prefix = prefix
        self._vlo = None
        self._d_kd = None
        self._pt_kd = None
        self.scene = scene

    @property
    def vlo(self):
        """direction vector (3,) luminance (srcn,), omega (1,)"""
        return self._vlo

    @property
    def d_kd(self):
        """list of direction kdtrees

        :getter: Returns kd tree structure
        :type: list of scipy.spatial.cKDTree
        """
        return self._d_kd

    @property
    def pt_kd(self):
        """point kdtree

        :getter: Returns kd tree structure
        :setter: Set this integrator's kd tree and scene data
        :type: scipy.spatial.cKDTree
        """
        return self._pt_kd

    def query(self, *args, **kwargs):
        """gather all rays from a point within a view cone"""
        return None, None, None

    def direct_view(self, vpts):
        """create a summary image of lightfield for each vpt"""
        pass
