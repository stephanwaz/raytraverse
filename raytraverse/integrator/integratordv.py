# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import translate
from raytraverse.integrator.integrator import Integrator
import raytraverse.integrator._helpers as intg
from raytraverse.lightpoint import LightPointKD


def _prep_dv(lpts, skyvecs, skpatch, skdir, side):
    skp = lpts[0]
    skdir = skdir.reshape(-1, 3)
    try:
        refl = np.loadtxt(f"{skp.scene.outdir}/{skp.parent}/"
                          f"reflection_normals.txt")
    except OSError:
        refl = []
    else:
        refl = translate.norm(refl.reshape(-1, 3))
        sunr = translate.reflect(skdir, refl, True)
        if len(sunr > 0):
            skdir = np.vstack((skdir, sunr))
    skvec = skp.vec
    sunlum = np.zeros((len(skp.lum), 2))
    spect = []
    vr = []

    for i, skd in enumerate(skdir):
        rpatch = translate.xyz2skybin(skd, side)
        cosa = translate.ctheta(skd, skp.vec)
        ang = cosa > np.cos(np.pi/side)
        np.set_printoptions(3, suppress=True)
        mv = np.max(skp.lum[ang, skpatch])
        spt = np.logical_and(skp.lum[:, skpatch] > mv/3, ang)
        opths = translate.xyz2skybin(skp.vec[spt], side) != rpatch
        if mv > .001 and np.sum(opths) > 0:
            # add cushion around source sampling
            vr.append(.533 + np.arccos(np.min(cosa[spt][opths])) * 180/np.pi)
        elif i == 0:
            vr.append(.533)
        else:
            vr.append(.533*2)
        spect.append(spt)

    # for i, skd in enumerate(skdir):
    #     ang = translate.ctheta(skd, skp.vec) > np.cos(np.pi/side)
    #     spect.append(np.logical_and(skp.lum[:, skpatch] > .005, ang))
    #     expoga = np.pi*2/(side*side)
    #     actoga = np.sum(skp.omega[spect[-1]])
    #     if actoga > 1.1*expoga:
    #         # add cushion around source sampling
    #         vr.append(.533 + 360/np.pi*(np.sqrt(actoga) -
    #                                     np.sqrt(expoga)))
    #     elif i == 0:
    #         vr.append(.533)
    #     else:
    #         vr.append(.533*2)
    # print(vr)
    nspect = np.logical_not(np.any(spect, axis=0))
    sunlum[nspect, 0] = skp.lum[nspect, skpatch]
    sklum = np.hstack((skp.lum, sunlum))
    srcn = skp.srcn+2
    if len(skyvecs[0]) < sklum.shape[1]:
        sklum = np.einsum('ij,kj->ki', skyvecs[0], sklum)
        sklum = np.hstack((sklum, np.zeros((len(sklum), 1))))
        srcn = len(skyvecs[0]) + 1
        skyvecs = [np.hstack((np.eye(len(skyvecs[0])), skyvecs[0][:, -1:]))]
    ski = LightPointKD(skp.scene, vec=skvec, lum=sklum, vm=skp.vm,
                       pt=skp.pt, posidx=skp.posidx,
                       src=f'sky_isun{skpatch:04d}', srcn=srcn,
                       srcdir=skdir[0],
                       write=False, omega=skp.omega, parent=skp.parent)
    lpts = [ski]
    return lpts, skyvecs, refl, vr


def evaluate_pt_dv(lpts, skyvecs, suns, **kwargs):
    side = int((skyvecs[0].shape[1] - 1)**.5)
    skpatch = translate.xyz2skybin(suns[0], side)[0]
    skdir = translate.skybin2xyz(skpatch, side)[0]
    lpts, skyvecs, refl, vr = _prep_dv(lpts, skyvecs, skpatch, skdir, side)
    return intg.evaluate_pt(lpts, skyvecs, suns, refl=refl, reflarea=vr, **kwargs)


def img_pt_dv(lpts, skyvecs, suns, **kwargs):
    side = int((skyvecs[0].shape[1] - 1)**.5)
    skpatch = translate.xyz2skybin(suns[0], side)[0]
    skdir = translate.skybin2xyz(skpatch, side)[0]
    lpts, skyvecs, refl, vr = _prep_dv(lpts, skyvecs, skpatch, skdir, side)
    return intg.img_pt(lpts, skyvecs, suns, refl=refl, reflarea=vr, **kwargs)


class IntegratorDV(Integrator):
    """specialized integrator for 2-phase Direct Viiews style calculation. assumes
    first lightplane is sky contrribution, second, direct sky contribution. Uses
    special point functions that combine two sky functions on a per patch basis.

    Parameters
    ----------
    skplane: raytraverse.lightfield.LightPlaneKD
    dskplane: raytraverse.lightfield.LightPlaneKD
    """
    evaluate_pt = evaluate_pt_dv
    img_pt = img_pt_dv

    def __init__(self, skplane, sunviewengine):
        super(IntegratorDV, self).__init__(skplane, sunviewengine=sunviewengine)

    def _group_query(self, skydata, points):
        # query and group sun positions
        gshape = (len(skydata.maskindices), len(points))
        suns = skydata.sun[:, None, 0:3]
        pts = np.atleast_2d(points)[None, :]
        vecs = np.concatenate(np.broadcast_arrays(suns, pts), -1).reshape(-1, 6)
        skydatas, dsns = self._unroll_sky_grid(skydata, gshape)
        idxs = []
        for lp in self.lightplanes:
            if lp.vecs.shape[1] == 6:
                idxs.append(lp.query(vecs)[0])
            else:
                idx, err = lp.query(points)
                idxs.append(np.broadcast_to(idx, gshape).ravel())
        idxs.append(np.broadcast_to(skydata.sunproxy[:, None], gshape).ravel())
        return np.stack(idxs), skydatas, dsns, vecs

    def _unroll_sky_grid(self, skydata, oshape):
        """

        Parameters
        ----------
        skydata: raytraverse.sky.SkyData
        oshape

        Returns
        -------

        """
        # broadcast skydata to match full indexing
        s = (*oshape[0:2], skydata.smtx.shape[1])
        smtx = np.broadcast_to(skydata.smtx[:, None, :], s).reshape(-1, s[-1])
        ds = (*oshape[0:2], skydata.sun.shape[1])
        dsns = np.broadcast_to(skydata.sun[:, None, :], ds).reshape(-1, ds[-1])
        # smtx (with sun), (patch sun, sun)
        skydatas = [np.hstack((smtx, dsns[:, -1:-3:-1]))]
        return skydatas, dsns
