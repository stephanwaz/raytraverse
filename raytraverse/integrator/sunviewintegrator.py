# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree, SphericalVoronoi
import clasp.script_tools as cst
from raytraverse import translate, io, optic
from raytraverse.mapper import ViewMapper
from raytraverse.integrator import Integrator


class SunViewIntegrator(object):
    """loads scene and sampling data for processing

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        prefix of data files to integrate
    rebuild: bool, optional
        build kd-tree even if one exists
    """

    def __init__(self, scene, suns, rebuild=False):
        #: np.array: sun positions
        self.suns = suns.suns
        #: raytraverse.sunmapper.SunMapper
        self.sunmap = suns.map
        #: str: prefix
        self.prefix = 'sunview'
        #: bool: force rebuild kd-tree
        self.rebuild = rebuild
        self.scene = scene

    @property
    def scene(self):
        """scene information

        :getter: Returns this integrator's scene
        :setter: Set this integrator's scene
        :type: raytraverse.scene.Scene
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        """Set this integrator's scene and load samples"""
        self._scene = scene
        dfile = f'{self.scene.outdir}/{self.prefix}_result.pickle'
        if not os.path.isfile(dfile):
            raise FileNotFoundError("No results files found, have you run"
                                    f" a Sampler of type {self.prefix} for"
                                    f" scene {self.scene.outdir}?")
        f = open(dfile, 'rb')
        self._vec = pickle.load(f)
        self._lum = pickle.load(f)
        self._omega = ((self.sunmap.viewangle/360)**2*np.pi**3)*pickle.load(f)
        f.close()
        kdfile = f'{self.scene.outdir}/{self.prefix}_kd_data.pickle'
        if os.path.isfile(kdfile) and not self.rebuild:
            f = open(kdfile, 'rb')
            self._pt_kd = pickle.load(f)
            self._sun_kd = pickle.load(f)
            f.close()
        else:
            self._pt_kd = cKDTree(self.scene.pts())
            self._sun_kd = cKDTree(self.suns)
            f = open(kdfile, 'wb')
            pickle.dump(self.pt_kd, f, protocol=4)
            pickle.dump(self.sun_kd, f, protocol=4)
            f.close()

    @property
    def lum(self):
        """luminance arrays grouped by point

        :getter: Returns luminance array
        :type: list of np.array
        """
        return self._lum

    @property
    def omega(self):
        """solid angles grouped by point

        :getter: Returns solid angles
        :type: list of np.array
        """
        return self._omega

    @property
    def vec(self):
        """direction vectors grouped by point

        :getter: Returns vector array
        :type: list of np.array
        """
        return self._vec

    @property
    def sun_kd(self):
        """list of direction kdtrees

        :getter: Returns kd tree structure
        :type: list of scipy.spatial.cKDTree
        """
        return self._sun_kd

    @property
    def pt_kd(self):
        """point kdtree

        :getter: Returns kd tree structure
        :setter: Set this integrator's kd tree and scene data
        :type: scipy.spatial.cKDTree
        """
        return self._pt_kd

    def query(self, vpts, suns):
        """gather all rays from a point within a view cone

        Parameters
        ----------
        vpts: np.array
            points to search for (broadcastable to directions)
        suns: np.array
            sun positions to search for (broadcastable to directions)

        Returns
        -------
        perrs: list or np.array
            for each of vpts, the distance to the returned values
        pis: list np.array
            for each of vpts, the index of the returned point
        idxs: list of list of np.array
            for each of vpts, for each vdir, an array of all idxs in
            self.kd.data matching the query

        """
        with ProcessPoolExecutor() as exc:
            perrs, pis = zip(*exc.map(self.pt_kd.query, vpts))
            serrs, sis = zip(*exc.map(self.sun_kd.query, suns))
        return perrs, pis, serrs, sis

    def apply_coefs(self, pis, sis, coefs):
        lum = []
        idx = []
        coefs = np.broadcast_to(coefs, len(sis))
        for pi in pis:
            for i, si in enumerate(sis):
                idx.append((pi, si))
                lum.append(self.lum[pi][si] * coefs[i])
        return lum, idx

    def view(self, vpts, suns, vdirs, decades=4, maxl=0.0, coefs=1.0,
             viewangle=180.0, **kwargs):
        """generate angular fisheye falsecolor luminance views
        additional kwargs passed to io.mk_img

        Parameters
        ----------
        vpts: np.array
            points to search for (broadcastable to directions)
        suns: np.array
            sun positions to search for (broadcastable to directions)
        vdirs: np.array
            directions to search for
        decades: real, optional
            number of log decades below max for minimum of color scale
        maxl: real, optional
            maximum log10(lum/179) for color scale
        coefs: np.array
            array of (N,) where N is the number of suns
            or float (applies to all suns)
        viewangle: float, optional
            degree opening of view cone

        Returns
        -------

        """
        vm = ViewMapper(viewangle=viewangle)
        ext = translate.xyz2xy(translate.tp2xyz(np.array([[viewangle*np.pi/360,
                                                           np.pi]])))[0, 0]
        perrs, pis, serrs, sis = self.query(vpts, suns)
        lums, idx = self.apply_coefs(pis, sis, coefs)
        viz = []
        vdirs = translate.norm(vdirs)
        for vd in vdirs:
            viz.append(np.arccos(np.dot(self.suns[np.array(sis)], vd))*180/np.pi
                       < viewangle/2)
        lums = [np.log10(l) for l in lums]
        futures = []
        func = io.mk_img_scatter
        with ProcessPoolExecutor() as exc:
            for k, v in enumerate(vdirs):
                vm.dxyz = v
                vz = viz[k]
                for (i, j), lum, vi in zip(idx, lums, vz):
                    pt = self.scene.idx2pt([i])[0]
                    if len(lum) == 0 or not vi:
                        print(f'Warning: Sun not visible from point: {pt} '
                              f'direction: {v}')
                    else:
                        vec = vm.xyz2xy(self.vec[i][j])
                        kw = dict(decades=decades, maxl=maxl, title=f"{pt} {v}",
                                  ext=ext, radius=self.sunmap.viewangle/64**2,
                                  **kwargs)
                        futures.append(exc.submit(func, lum, vec,
                                       f'{self.prefix}_{i}_{k}_{j}.pdf', **kw))
        return [fut.result() for fut in futures]

    def view_together(self, vpts, suns, vdirs, decades=4, maxl=0.0, coefs=1.0,
                      viewangle=180.0, **kwargs):
        """generate angular fisheye falsecolor luminance views
        additional kwargs passed to io.mk_img

        Parameters
        ----------
        vpts: np.array
            points to search for (broadcastable to directions)
        suns: np.array
            sun positions to search for (broadcastable to directions)
        vdirs: np.array
            directions to search for
        decades: real, optional
            number of log decades below max for minimum of color scale
        maxl: real, optional
            maximum log10(lum/179) for color scale
        coefs: np.array
            array of (N,) where N is the number of suns
            or float (applies to all suns)
        viewangle: float, optional
            degree opening of view cone

        Returns
        -------

        """
        vm = ViewMapper(viewangle=viewangle)
        ext = translate.xyz2xy(translate.tp2xyz(np.array([[viewangle*np.pi/360,
                                                           np.pi]])))[0, 0]
        perrs, pis, serrs, sis = self.query(vpts, suns)
        lums, idx = self.apply_coefs(pis, sis, coefs)
        viz = []
        vdirs = translate.norm(vdirs)
        for vd in vdirs:
            viz.append(np.arccos(np.dot(self.suns[np.array(sis)], vd))*180/np.pi
                       < viewangle/2)
        lums = [np.log10(l) for l in lums]
        for k, v in enumerate(vdirs):
            vm.dxyz = v
            vz = viz[k]
            avecs = []
            alums = []
            for (i, j), lum, vi in zip(idx, lums, vz):
                pt = self.scene.idx2pt([i])[0]
                if len(lum) == 0 or not vi:
                    print(f'Warning: Sun not visible from point: {pt} '
                          f'direction: {v}')
                else:
                    avecs.append(vm.xyz2xy(self.vec[i][j]))
                    alums.append(lum)
            kw = dict(decades=decades, maxl=maxl, title=f"{pt} {v}",
                      ext=ext, radius=self.sunmap.viewangle/64**2,
                      **kwargs)
            lum = np.concatenate(alums)
            vec = np.concatenate(avecs)
            io.mk_img_scatter(lum, vec, f'{self.prefix}_{k}.png', **kw)
        # return [fut.result() for fut in futures]
