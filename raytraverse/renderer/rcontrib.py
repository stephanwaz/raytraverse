# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os

from clasp import script_tools as cst

from raytraverse import io
from craytraverse.renderer import Rcontrib as pRcontrib
from raytraverse.formatter import RadianceFormatter as Fmt
from craytraverse.renderer.rcontrib import rcontrib_instance


class Rcontrib(pRcontrib):
    """singleton wrapper for c++ raytrraverse.crenderer.cRcontrib class

    this class sets default arguments, helps with initialization and setting
    cpu limits of the cRcontrib instance. see raytrraverse.crenderer.cRcontrib
    for more details.

    Parameters
    ----------
    rayargs: str, optional
        argument string (options and flags only) raises ValueError if arguments
        are not recognized by cRtrace.
    scene: str, optional
        path to octree
    nproc: int, optional
        if None, sets nproc to cpu count, or the RAYTRAVERSE_PROC_CAP
        environment variable
    skyres: int, optional
        resolution of sky patches (sqrt(patches / hemisphere)).
        So if skyres=18, each patch
        will be 100 sq. degrees (0.03046174197 steradians) and there will be
        18 * 18 = 324 sky patches.
    modname: str, optional
        passed the -m option of cRcontrib initialization
    ground: bool, optional
        if True include a ground source (included as a final bin)
    default_args: bool, optional
        if True, prepend default args to rayargs parameter
    adpatch: int, optional
            when using default_args, ad is set to this times srcn

    Examples
    --------

    Basic Initialization and call::

        r = renderer.Rcontrib(args, scene)
        ans = r(vecs)
        # ans.shape -> (vecs.shape[0], 325)
    """
    instance = rcontrib_instance
    adpatch = 50
    nproc = None

    def __init__(self, rayargs=None, scene=None, nproc=None,
                 skyres=15, modname='skyglow', ground=True,
                 default_args=True, adpatch=50):
        Rcontrib.adpatch = adpatch
        if default_args:
            if rayargs is None:
                rayargs = self.get_default_args()
            else:
                rayargs = f"{self.get_default_args()} {rayargs}"
        super().__init__(rayargs, scene, nproc=nproc, skyres=skyres,
                         modname=modname, ground=ground)

    @classmethod
    def set_args(cls, args, nproc=None):
        """prepare arguments to call engine instance initialization

        Parameters
        ----------
        args: str
            rendering options
        nproc: int, optional
            cpu limit

        """
        if nproc is None:
            nproc = cls.nproc
        nproc = io.get_nproc(nproc)
        if "-ab 0" in args:
            nproc = 1
        super().set_args(args, nproc)

    @classmethod
    def setup(cls, scene=None, ground=True, modname="skyglow", skyres=15):
        """set class attributes for proper argument initialization

        Parameters
        ----------
        scene: str, optional
            path to octree
        ground: bool, optional
            if True include a ground source (included as a final bin)
        modname: str, optional
            passed the -m option of cRcontrib initialization
        skyres: float, optional
            resolution of sky patches (sqrt(patches / hemisphere)).
            So if skyres=10, each patch will be 100 sq. degrees
            (0.03046174197 steradians) and there will be 18 * 18 = 324 sky
            patches.

        Returns
        -------
        scene: str
            path to scene with added sky definition

        """
        cls.ground = ground
        if scene is not None:
            srcdef = Fmt.get_skydef((1, 1, 1), ground=ground, name=modname)
            ocom = f'oconv -f -i {scene} -'
            scene = scene.rsplit(".", 1)[0] + "_sky.oct"
            if not os.path.isfile(scene):
                f = open(scene, 'wb')
                cst.pipeline([ocom], outfile=f, inp=srcdef, close=True)
        cls.skyres = skyres
        cls.srcn = cls.skyres**2 + ground
        cls.modname = modname
        return scene

    @classmethod
    def get_default_args(cls):
        """construct default arguments"""
        return ("-u+ -ab 16 -av 0 0 0 -aa 0 -as 0 -dc 1 -dt 0 -lr -14 -ad "
                f"{cls.adpatch*cls.srcn} -lw {0.4/(cls.srcn*cls.adpatch)} "
                "-st 0 -ss 16 -c 1")
