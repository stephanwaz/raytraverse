# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================


"""factory functions for easy api access raytraverse."""
import os
from glob import glob

from raytraverse.scene import Scene
from raytraverse.sky import SkyData
from raytraverse.mapper import PlanMapper
from raytraverse.lightfield import DayLightPlaneKD


def auto_reload(scndir, area, areaname="plan", skydata="skydata", ptres=1.0,
                rotation=0.0, zheight=None, includesky=True):
    """reload daylight plane and associated class instances from file paths

    Parameters
    ----------
    scndir: str
        matches outdir argument of Scene()
    area: str np.array, optional
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
    includesky: bool, optional
        include lightplane for sky source with DaylightPlaneKD for suns

    Returns
    -------
    DayLightPlaneKD
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
    try:
        sunfile = glob(f"{scndir}/{areaname}/*_sunpositions.tsv")[0]
    except IndexError:
        raise FileNotFoundError(f"no sunpositions file in {scndir}/{areaname}")
    skname = sunfile.split("/")[-1][:-13]
    if os.path.exists(f"{scndir}/{areaname}/i_{skname}_0000"):
        skname = f"i_{skname}"
    lp = DayLightPlaneKD(scn, sunfile, pm, skname, includesky=includesky)
    return lp, skd
