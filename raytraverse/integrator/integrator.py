# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree, SphericalVoronoi
import clasp.script_tools as cst
from raytraverse import translate, io, optic
from raytraverse.lightfield import SunField, SrcBinField
from raytraverse.mapper import ViewMapper

class Integrator(object):
    """loads scene and sampling data for processing

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        sun class containing sun vectors
    """

    def __init__(self, scene, suns, stol=10.0):
        #: raytraverse.scene.Scene
        self.scene = scene
        #: raytraverse.sunsetter.SunSetter
        self.suns = suns
        self.stol = stol
        self.sunfield = SunField(scene, suns)
        self.skyfield = SrcBinField(scene, prefix='sky')
        self.dayhours = self.scene.skydata[:, 0] > 0
        xyz = translate.aa2xyz(self.scene.skydata[self.dayhours, 0:2])
        self.hassun = self.sunfield.has_proxy_src(xyz, tol=self.stol)

    def get_sky_commands(self):
        sd = self.scene.skydata[self.dayhours]
        sc = 'gendaylit -ang {:0= 9.4f} {:0= 9.4f} -W {:0= 8.2f} {:0= 8.2f} {}'
        flag = np.full(sd.shape[0], '  ', dtype='U2')
        flag[self.hassun] = '-s'
        skycoms = np.empty(sd.shape[0], 'U58')
        for i in range(sd.shape[0]):
            skycoms[i] = sc.format(*sd[i], flag[i])
        return skycoms

    def get_sky_mtx(self):
        skycoms = self.get_sky_commands()
        at = []
        sc = np.random.choice(skycoms, 30)
        # for i in sc:
        #     a = cst.pipeline([i,])
        #     at.append([float(j) for j in a.strip().split()[-10:]])
        print(sc)
        print(at)
        # gsv = np.full(1, f'genskyvec_sc -sc -m {self.scene.skyres} -h -1 -b')
        # gsvcoms = np.broadcast_to(gsv, skycoms.shape)
        # with ThreadPoolExecutor() as ex:
        #     cols = ex.map(io.call_generic, zip(skycoms[0:1000], gsvcoms))
        # smtx = np.hstack(cols)
        # print(smtx.shape)

    # def apply_coefs(self, pis, coefs=None):
    #     cnt = self.lum[pis[0]].shape[1]
    #     if coefs is None:
    #         lum = [np.sum(self.lum[pi], 1).reshape(1, -1) for pi in pis]
    #     elif coefs.shape[-1] == cnt:
    #         skyvecs = coefs.reshape(-1, cnt).T
    #         lum = [(self.lum[pi].reshape(-1, cnt)@skyvecs).T for pi in pis]
    #     else:
    #         coefs = coefs.reshape(-1, 2).T
    #         bins = coefs[0].astype(int)
    #         c = coefs[1]
    #         lum = [(self.lum[pi].reshape(-1, cnt)[:, bins]*c).T for pi in pis]
    #     return lum
    #
    # def view(self, vpts, vdirs, decades=4, maxl=0.0, coefs=None, treecnt=30,
    #          ring=150, viewangle=180.0, ringtol=15.0, scatter=False, **kwargs):
    #     """generate angular fisheye falsecolor luminance views
    #     additional kwargs passed to io.mk_img
    #
    #     Parameters
    #     ----------
    #     vpts: np.array
    #         points to search for (broadcastable to directions)
    #     vdirs: np.array
    #         directions to search for
    #     decades: real, optional
    #         number of log decades below max for minimum of color scale
    #     maxl: real, optional
    #         maximum log10(lum/179) for color scale
    #     coefs: np.array
    #         array of (N,) + lum.shape[1:] (coefficient vector)
    #         or array of (N, 2) (index and coefficient)
    #     treecnt: int, optional
    #         number of queries at which a scipy.cKDtree.query_ball_tree is
    #         used instead of scipy.cKDtree.query_ball_point
    #     ring: int, optional
    #         add points around perimeter to clean up edge, set to 0 to disable
    #     viewangle: float, optional
    #         degree opening of view cone
    #     ringtol: float, optional
    #         tolerance (in degrees) for adding ring points
    #
    #     Returns
    #     -------
    #
    #     """
    #     popts = np.get_printoptions()
    #     np.set_printoptions(2)
    #     rt = translate.theta2chord(ringtol/180*np.pi)
    #     vm = ViewMapper(viewangle=viewangle)
    #     ext = translate.xyz2xy(translate.tp2xyz(np.array([[viewangle*np.pi/360,
    #                                                        np.pi]])))[0, 0]
    #     perrs, pis, idxs = self.query(vpts, vdirs, treecnt=treecnt,
    #                                   viewangle=viewangle)
    #     lum = self.apply_coefs(pis, coefs=coefs)
    #     lum = [np.log10(l) for l in lum]
    #     rvecs, rxy = translate.mkring(viewangle, ring)
    #     futures = []
    #     if scatter:
    #         func = io.mk_img_scatter
    #     else:
    #         func = io.mk_img
    #     with ProcessPoolExecutor() as exc:
    #         for k, v in enumerate(vdirs):
    #             vm.dxyz = v
    #             trvecs = vm.view2world(rvecs)
    #             for li, (perr, pi, idx) in enumerate(zip(perrs, pis, idxs)):
    #                 pt = self.scene.idx2pt([pi])[0]
    #                 if len(idx[k]) == 0:
    #                     print(f'Warning: No rays found at point: {pt} '
    #                           f'direction: {v}')
    #                 else:
    #                     vec = vm.xyz2xy(self.vec[pi][idx[k]])
    #                     d, tri = self.d_kd[pi].query(trvecs,
    #                                                  distance_upper_bound=rt)
    #                     vec = np.vstack((vec, rxy[d < ringtol]))
    #                     vi = np.concatenate((idx[k], tri[d < ringtol]))
    #                     for j in range(lum[li].shape[0]):
    #                         kw = dict(decades=decades, maxl=maxl,
    #                                   inclmarks=len(idx[k]), title=f"{pt} {v}",
    #                                   ext=ext, **kwargs)
    #                         futures.append(exc.submit(func, lum[li][j, vi],
    #                                                   vec, f'{self.prefix}_{pi}_{k}_{j}.png', **kw))
    #     np.set_printoptions(**popts)
    #     return [fut.result() for fut in futures]
    #
    # def voronoi(self, vpts, vdirs, decades=4, maxl=0.0, coefs=None, treecnt=30,
    #             viewangle=180.0, **kwargs):
    #     """generate angular fisheye falsecolor luminance views
    #     additional kwargs passed to io.mk_img
    #
    #     Parameters
    #     ----------
    #     vpts: np.array
    #         points to search for (broadcastable to directions)
    #     vdirs: np.array
    #         directions to search for
    #     decades: real, optional
    #         number of log decades below max for minimum of color scale
    #     maxl: real, optional
    #         maximum log10(lum/179) for color scale
    #     coefs: np.array
    #         array of (N,) + lum.shape[1:] (coefficient vector)
    #         or array of (N, 2) (index and coefficient)
    #     treecnt: int, optional
    #         number of queries at which a scipy.cKDtree.query_ball_tree is
    #         used instead of scipy.cKDtree.query_ball_point
    #     viewangle: float, optional
    #         degree opening of view cone
    #
    #     Returns
    #     -------
    #
    #     """
    #     popts = np.get_printoptions()
    #     np.set_printoptions(2)
    #     vm = ViewMapper(viewangle=viewangle)
    #     ext = translate.xyz2xy(translate.tp2xyz(np.array([[viewangle*np.pi/360,
    #                                                        np.pi]])))[0, 0]
    #     perrs, pis, idxs = self.query(vpts, vdirs, treecnt=treecnt,
    #                                   viewangle=viewangle)
    #     lum = self.apply_coefs(pis, coefs=coefs)
    #     lum = [np.log10(l) for l in lum]
    #     futures = []
    #     with ProcessPoolExecutor() as exc:
    #         for k, v in enumerate(vdirs):
    #             vm.dxyz = v
    #             for li, (perr, pi, idx) in enumerate(zip(perrs, pis, idxs)):
    #                 pt = self.scene.idx2pt([pi])[0]
    #                 if len(idx[k]) == 0:
    #                     print(f'Warning: No rays found at point: {pt} '
    #                           f'direction: {v}')
    #                 else:
    #                     vi = idx[k]
    #                     vec = vm.xyz2xy(self.vec[pi][vi])
    #                     verts = vm.xyz2xy(self.svs[pi].vertices)
    #                     regions = self.svs[pi].regions
    #                     for j in range(lum[li].shape[0]):
    #                         kw = dict(decades=decades, maxl=maxl,
    #                                   title=f"{pt} {v}", ext=ext,
    #                                   outf=f'{self.prefix}_{pi}_{k}_{j}.png',
    #                                   **kwargs)
    #                         futures.append(
    #                             exc.submit(io.mk_img_voronoi, lum[li][j],
    #                                        vec, verts, regions, vi, **kw))
    #     np.set_printoptions(**popts)
    #     return [fut.result() for fut in futures]
    #
    # def illum(self, vpts, vdirs, coefs=None, treecnt=30):
    #     """calculate illuminance for given sensor locations and skyvecs
    #
    #     Parameters
    #     ----------
    #     vpts: np.array
    #         points to search for (broadcastable to directions)
    #     vdirs: np.array
    #         directions to search for
    #     coefs: np.array
    #         array of (N,) + lum.shape[1:] (coefficient vector)
    #         or array of (N, 2) (index and coefficient)
    #     treecnt: int, optional
    #         number of queries at which a scipy.cKDtree.query_ball_tree is
    #         used instead of scipy.cKDtree.query_ball_point
    #
    #     Returns
    #     -------
    #     illum: np.array (vdirs.shape[0], skyvecs.shape[0])
    #         illuminance at each point/direction and sky weighting
    #     """
    #     perrs, pis, idxs = self.query(vpts, vdirs, treecnt=treecnt)
    #     vdirs = translate.norm(vdirs)
    #     lum = self.apply_coefs(pis, coefs=coefs)
    #     futures = []
    #     with ProcessPoolExecutor() as exc:
    #         for j in range(lum[0].shape[0]):
    #             for li, (perr, pi, idx) in enumerate(zip(perrs, pis, idxs)):
    #                 for k, v in enumerate(vdirs):
    #                     vec = self.vec[pi][idx[k]]
    #                     omega = self.omega[pi][idx[k]]
    #                     futures.append(exc.submit(optic.calc_illum, v, vec.T,
    #                                               omega, lum[li][j, idx[k]]))
    #     return np.array([fut.result()
    #                      for fut in futures]).reshape((lum[0].shape[0],
    #                                                    len(pis), len(vdirs)))
