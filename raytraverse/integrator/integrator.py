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
from raytraverse import translate, io, optic, skycalc
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

    def get_sky_mtx(self):
        sxyz = translate.aa2xyz(self.scene.skydata[self.dayhours, 0:2])
        hassun, si = self.suns.proxy_src(sxyz, tol=self.stol)
        nosun = np.arange(hassun.size)[np.logical_not(hassun)]
        sunuv = translate.xyz2uv(sxyz[nosun], flipu=False)
        sunbins = translate.uv2bin(sunuv, self.scene.skyres)
        dirdif = self.scene.skydata[self.dayhours, 2:]
        smtx, grnd, sun = skycalc.sky_mtx(sxyz, dirdif, self.scene.skyres)
        # ratio between actual solar disc and patch
        omegar = np.square(0.2665 * np.pi * self.scene.skyres / 180) * .5
        plum = sun[nosun] * omegar
        smtx[nosun, sunbins] += plum
        oor = len(si)
        return smtx, grnd, sun, np.where(hassun, si, oor)

    def hdr(self, pts, vdir, smtx, suns, hassun,
            vname='view', viewangle=180.0, res=400, interp=1):
        """

        Parameters
        ----------
        pts: np.array
            points
        vdir: (float, float, float)
            view direction for images
        smtx: np.array
            sky matrix
        suns: np.array
            sun values
        hassun: np.array
            boolean array if a high res sun exists
        vname: str
            view name for output file
        viewangle: float, optional
            degree opening of view cone
        res: int, optional
            image resolution
        interp: int, optional
            number of nearest points to interpolate between. 1 will resemble
            voronoi patches

        Returns
        -------

        """
        perrs, pis = self.scene.pt_kd.query(pts)
        vm = ViewMapper(viewangle=viewangle, dxyz=vdir)
        pdirs, mask = vm.pixelrays(res)
        img = np.zeros((res, res))
        for pi in pis:
            skylum = self.skyfield.measure(pi, pdirs[mask], smtx, interp=interp)
            for sj, skyv in enumerate(skylum):
                outf = f"{self.scene.outdir}_{vname}_{pi:04d}_{sj:04d}.hdr"
                img[mask] = skyv
                if hassun[sj] < self.suns.suns.shape[0] and suns[sj] > 0:
                    psi = (pi, hassun[sj])
                    img[mask] += self.sunfield.measure(psi, pdirs[mask],
                                                       suns[sj], interp=interp)
                    spix, svals = self.sunfield.draw_sun(psi, suns[sj], vm, res)
                    if spix is not None:
                        print(outf)
                        img[spix[:, 0], spix[:, 1]] += svals
                io.array2hdr(img, outf)






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
