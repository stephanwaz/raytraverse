# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from raytraverse.renderer.radiancerenderer import RadianceRenderer
from raytraverse.crenderer import cRtrace


class Rtrace(RadianceRenderer):
    """singleton wrapper for c++ raytrraverse.crenderer.cRtrace class

    this class sets default arguments, helps with initialization and setting
    cpu limits of the cRtrace instance. see raytrraverse.crenderer.cRtrace
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
    default_args: bool, optional
        if True, prepend default args to rayargs parameter
    direct: bool, optional
        if True use Rtrace.directargs in place of default (also if True, sets
        default_args to True.

    Examples
    --------

    Basic Initialization and call::

        r = renderer.Rtrace(args, scene)
        ans = r(vecs)
        # ans.shape -> (vecs.shape[0], 1)

    """
    name = 'rtrace'
    #: raytraverse.crenderer.cRtrace
    engine = cRtrace
    defaultargs = "-av 0 0 0 -aa 0 -ab 7 -ad 128 -as 0 -c 10 -as 0 -lw 1e-5"
    directargs = "-av 0 0 0 -ab 0 -lr 0"
    usedirect = False

    def __init__(self, rayargs=None, scene=None, nproc=None,
                 default_args=True, direct=False):
        type(self).usedirect = direct
        default_args = default_args or direct
        super().__init__(rayargs, scene, nproc, default_args=default_args)

    @classmethod
    def get_default_args(cls):
        """return default arguments of the class"""
        if cls.usedirect:
            return cls.directargs
        else:
            return cls.defaultargs

    @classmethod
    def update_ospec(cls, vs):
        """set output of cRtrace instance

        Parameters
        ----------
        vs: str
            output specifiers for rtrace::
                o    origin (input)
                d    direction (normalized)
                v    value (radiance)
                V    contribution (radiance)
                w    weight
                W    color coefficient
                l    effective length of ray
                L    first intersection distance
                c    local (u,v) coordinates
                p    point of intersection
                n    normal at intersection (perturbed)
                N    normal at intersection (unperturbed)
                r    mirrored value contribution
                x    unmirrored value contribution
                R    mirrored ray length
                X    unmirrored ray length

        Returns
        -------
        outcnt: int
            the number of output columns to expect when calling rtrace instance

        Raises
        ------
        ValueError:
            when an output specifier is not recognized
        """
        outcnt = cls.instance.update_ospec(vs)
        if outcnt < 0:
            raise ValueError(f"Could not update {cls.__name__} with "
                             f"outputs: '{vs}'")
        return outcnt

    @classmethod
    def load_source(cls, srcname, freesrc=-1):
        """add a source description to the loaded scene

        Parameters
        ----------
        srcname: str
            path to radiance scene file containing sources, these should not
            change the bounding box of the octree and has only been tested with
            the "source" type.
        freesrc: int, optional
            the number of objects to unload from the end of the rtrace object
            list, if -1 unloads all objects loaded by previous calls to
            load_source
        """
        cls.instance.load_source(srcname, freesrc)
