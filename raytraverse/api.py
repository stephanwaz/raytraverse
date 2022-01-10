# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================


"""factory functions for easy api access raytraverse."""
import os

import numpy as np

from raytraverse.integrator import IntegratorDS
from raytraverse.integrator.integrator import Integrator
from raytraverse.scene import Scene
from raytraverse.sky import SkyData
from raytraverse.mapper import PlanMapper
from raytraverse.lightfield import SunsPlaneKD, LightPlaneKD
from raytraverse.lightpoint import LightPointKD


def auto_reload(scndir, area, areaname="plan", skydata="skydata", ptres=1.0,
                rotation=0.0, zheight=None):
    """reload associated class instances from file paths

    Parameters
    ----------
    scndir: str
        matches outdir argument of Scene()
    area: str np.array
        radiance scene geometry defining a plane to sample, tsv file of
        points to generate bounding box, or np.array of points.
    areaname: str, optional
        matches name argument of PlanMapper()
    skydata: str, optional
        matches name argument of SkyData.write()
    ptres: float, optional
        resolution for considering points duplicates, border generation
        (1/2) and add_grid(). updateable
    rotation: float, optional
        positive Z rotation for point grid alignment
    zheight: float, optional
        override calculated zheight

    Returns
    -------
    Scene
    PlanMapper
    SkyData
    """
    if not os.path.exists(scndir):
        raise FileNotFoundError(f"auto_reload is only for reloading existing "
                                f"scenes. {scndir} does not exist")
    scn = Scene(scndir)
    if not os.path.isfile(f"{scndir}/{skydata}.npz"):
        raise FileNotFoundError(f"auto_reload is only for reloading existing "
                                f"skydata. {scndir}/{skydata}.npz does not "
                                f"exist")
    skd = SkyData(f"{scndir}/{skydata}.npz")
    pm = PlanMapper(area, name=areaname, ptres=ptres, rotation=rotation,
                    zheight=zheight)
    return scn, pm, skd


def load_lp(path, hasparent=True):
    ftree = path.split("/")
    if hasparent:
        scndir = ftree[-4]
        parent = ftree[-3]
    else:
        scndir = ftree[-3]
        parent = None
    scn = Scene(scndir)
    pidx = int(ftree[-1].split(".")[0])
    try:
        pts = np.loadtxt(path.replace(f"/{ftree[-1]}", "_points.tsv"))
    except FileNotFoundError:
        pt = (0, 0, 0)
    else:
        pt = pts[pidx, -3:]
    return LightPointKD(scn, parent=parent, src=ftree[-2], posidx=pidx, pt=pt)


stypes = ('1comp', '2comp', '3comp', 'direct', 'directpatch', 'sunonly',
          'sunpatch', 'skyonly')


def get_integrator(scn, pm, srcname="suns", simtype="2comp"):
    req_sun = ('2comp', '3comp', 'direct', 'directview')
    req_sky = ('1comp', '2comp', '3comp', 'sunpatch')
    req_dsk = ('3comp', 'directpatch')
    sunfile = f"{scn.outdir}/{pm.name}/{srcname}_sunpositions.tsv"
    skpoints = f"{scn.outdir}/{pm.name}/sky_points.tsv"
    dskpoints = f"{scn.outdir}/{pm.name}/skydcomp_points.tsv"

    try:
        sunplane = SunsPlaneKD(scn, sunfile, pm, f"{srcname}_sun")
    except OSError:
        if simtype in req_sun:
            raise OSError(f"file: {sunfile} does not exist, make sure that a"
                          f" complete sun sampling exists")
        sunplane = None
    try:
        skyplane = LightPlaneKD(scn, skpoints, pm, "sky")
    except OSError:
        if simtype in req_sky:
            raise OSError(f"file: {skpoints} does not exist, make sure that a"
                          f" complete sky sampling exists")
        skyplane = None
    try:
        dskplane = LightPlaneKD(scn, dskpoints, pm, "skydcomp")
    except OSError:
        if simtype in req_dsk:
            raise OSError(f"file: {dskpoints} does not exist, make sure that a"
                          f" complete direct sky sampling exists")
        dskplane = None

    if simtype in ["1comp", "sunpatch", "skyonly"]:
        return Integrator(skyplane, includesky=simtype != "sunpatch",
                          includesun=simtype != "skyonly")
    if simtype == "2comp":
        return Integrator(skyplane, sunplane)
    if simtype == "3comp":
        return IntegratorDS(skyplane, dskplane, sunplane)
    if simtype in ["direct", "sunonly"]:
        return Integrator(sunplane, includesky=False)
    if simtype == "directpatch":
        return Integrator(dskplane, includesky=False)
    raise ValueError(f"Error loading {simtype}")
