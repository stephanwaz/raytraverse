# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from scipy.spatial import cKDTree
from sklearn.cluster import Birch

from raytraverse import translate, io
from raytraverse.lightpoint.lightpointkd import LightPointKD


class CompressedPointKD(LightPointKD):
    """compressed data needs special methods for making images.

    can be initialized either like LightPointKD (but with required omega
    argument), or if 'scene' is a LightPointKD then a compressed output is
    calculated from the input

    Parameters
    ----------
    scene: BaseScene LightpointKD
    src: str, optional
        new name for src passed to LightPointKD constructor
    dist: float, optional
        translate.theta2chord(np.pi/32), primary clustering distance
        using the birch algorithm, for lossy compression of lf. this is the
        maximum radius of a cluster, preserving important directional
        information. clustering acts on ray direction and luminance, with
        weight of luminance dimension controlled by the lweight parameter.
    lerr: float, optional
        min-max normalized error in luminance grouping.
    plotc: bool, optional
        make directview plot of compressed output showing source vectors
    """

    def __init__(self, scene, vec=None, lum=None, write=True, src=None,
                 dist=0.0981, lerr=0.01, plotc=False, **kwargs):
        if issubclass(type(scene), LightPointKD):
            new = True
            if plotc:
                self._clusterimg = np.zeros((3, 512*scene.vm.aspect, 512))
            scn, kwargs = self.compress(scene, src=src, dist=dist, lerr=lerr)

        else:
            new = vec is not None and lum is not None
            scn = scene
            self._clusterimg = None
        kwargs.update(filterviews=False, calcomega=False, write=False)
        super().__init__(scn, **kwargs)
        if self.omega is None:
            raise ValueError(f"{type(self)} must be initialized with "
                             f"precomputed omega")
        # wait to write until success
        if write and new:
            self.dump()
        if self._clusterimg is not None:
            res = 1024
            vimg = translate.resample(self._clusterimg,
                                      (3, res * self.vm.aspect, res),
                                      gauss=False)
            img, pdirs, mask, mask2, header = self.vm.init_img(res, self.pt)
            self.add_to_img(img, pdirs[mask], mask)
            if mask2 is not None:
                self.add_to_img(img[res:], pdirs[res:][mask2], mask2)
            outf = self.file.replace("/", "_").replace(".rytpt", "_clust.hdr")
            vimg = np.where(np.sum(vimg, axis=0)[None] == 0, img[None], vimg)
            io.carray2hdr(vimg, outf, header=[header])
            self._clusterimg = None

    def add_to_img(self, img, vecs, mask=None, skyvec=1, vm=None, **kwargs):
        """add luminance contributions to image array (updates in place)

        Parameters
        ----------
        img: np.array
            2D image array to add to (either zeros or with other source)
        vecs: np.array
            vectors corresponding to img pixels shape (N, 3)
        mask: np.array, optional
            indices to img that correspond to vec (in case where whole image
            is not being updated, such as corners of fisheye)
        skyvec: int float np.array, optional
            source coefficients, shape is (1,) or (srcn,)
        vm: raytraverse.mapper.ViewMapper, optional
        """
        val = np.squeeze(self.apply_coef(skyvec))
        imgkd = cKDTree(vecs)
        r = translate.theta2chord(np.sqrt(self.omega/np.pi))
        splats = imgkd.query_ball_point(self.vec, r)
        img0 = np.zeros(len(vecs))
        for sp, lum in zip(splats, val):
            img0[sp] += lum
        img[mask] += img0
        for srcview in self.srcviews:
            srcview.add_to_img(img, vecs, mask, skyvec[-1], vm)

    def compress(self, lp, src=None, dist=0.0981, lerr=0.01):
        """A lossy compression based on clustering. Rays are clustered using
        the birch algoritm on a 4D vector  (x,y,z,lum) where lum is the sum
        of contributions from all sources in the LightPoint. In the optional
        second stage (activated with secondary=True) sources are further
        grouped through agglomerative cluster using an average linkage. this
        is to help with source indentification/matching between LightPoints,
        but can introduce significant errors to computing non energy
        conserving metrics in cases where the applied sky vectors have large
        relative differences between adjacent patches (> 1.5:1) or if the
        variance in peak luminance above the lthreshold parameter is
        significant. These include cases where nearby transmitting materials
        is varied (example: a trans upper above a clear lower), or lthreshold
        is set too low. For this reason, it is better to use single stage
        compression for metric computation and only do glare source grouping
        for interpolation between LightPoints.

        Parameters
        ----------
        lp: LightPointKD
        src: str, optional
            new name for src passed to LightPointKD constructor
        dist: float, optional
            translate.theta2chord(np.pi/32), primary clustering distance
            using the birch algorithm, for lossy compression of lf. this is the
            maximum radius of a cluster, preserving important directional
            information. clustering acts on ray direction and luminance, with
            weight of luminance dimension controlled by the lweight parameter.
        lerr: float, optional
            min-max normalized error in luminance grouping.
        plotc: bool, optional
            make directview plot of compressed output showing source vectors

        Returns
        -------
        arguments for initializing a CompressedPointKD

        """
        lweight = dist/lerr
        wv = self._source_weighted_vector(lp, lweight)
        clust = Birch(threshold=dist, n_clusters=None)
        clust.fit(wv)
        lsort = np.argsort(clust.labels_)
        ul, sidx = np.unique(clust.labels_[lsort], return_index=True)
        slices = np.array_split(lsort, sidx[1:])
        ovec, ooga, olum = self._reduce(lp, slices)
        if src is None:
            src = f"{lp.src}_compressed"
        kwargs = dict(vec=ovec, lum=olum, vm=lp.vm, pt=lp.pt, posidx=lp.posidx,
                      src=src, srcn=lp.srcn, omega=ooga, srcdir=lp.srcdir,
                      srcviews=lp.srcviews)
        return lp.scene, kwargs

    @staticmethod
    def _source_weighted_vector(lp, weight=10):
        """generate vector for clustering

        Parameters
        ----------
        lp: LightPointKD
        weight: float, optional
            coefficient for min-max normalized luminances, bigger values weight
            luminance more strongly compared to vector direction, meaning with
            higher numbers clusters will have less variance in luminance.

        Returns
        -------
        weighted_vector: np.array
            (N, 7) ray direction, source direction, and source brightness
        """
        lum = lp.apply_coef(1).T
        # min-max normalize luminances
        bound = np.percentile(lum, (0, 100))
        scale = bound[1] - bound[0]
        ldist = (lum - bound[0])/scale
        src = np.einsum('jk,ij->ik', lp.srcdir, lp.lum)
        return np.hstack((lp.vec, src, ldist * weight))

    def _reduce(self, lp, slices):
        """group vector/omega/lum data according to slices

        Parameters
        ----------
        lp: LightPointKD
        slices: list
            length = len(np.unique(labels))
            list of index arrays for each cluster label

        Returns
        -------
        vec: np.array
            shape (n_clust, 3), reduced vectors
        oga: np.array
            shape (n_clust,), reduced omeega
        lum: np.array
            shape (n_clust, nsrcs), reduced lum
        """
        v2 = []
        o2 = []
        l2 = []
        lum = np.reshape(lp.lum, (lp.vec.shape[0], -1))
        rng = np.random.default_rng()
        for s in slices:
            v = np.average(lp.vec[s], 0, weights=lp.omega[s])
            mag = np.linalg.norm(v)
            if self._clusterimg is not None:
                color = rng.random(3)
                color[rng.integers(3, size=1)] = 0
                self._clusterimg = lp.vm.add_vecs_to_img(self._clusterimg,
                                                         lp.vec[s],
                                                         channels=color)
            v2.append(v/mag)
            o2.append(np.sum(lp.omega[s]))
            l2.append(np.average(lum[s], 0, weights=lp.omega[s]))
        return np.array(v2), np.array(o2), np.array(l2)

