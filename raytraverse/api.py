# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================


"""factory functions for easy api access raytraverse."""
import os

from raytraverse import io
from raytraverse.integrator import Integrator
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
    if hasparent:
        try:
            ftree = path.rsplit("/", 3)
            scndir = ftree[-4]
            parent = ftree[-3]
        except IndexError:
            ftree = path.rsplit("/", 2)
            scndir = ftree[-3]
            parent = None
    else:
        ftree = path.rsplit("/", 2)
        scndir = ftree[-3]
        parent = None
    scn = Scene(scndir)
    pidx = int(ftree[-1].split(".")[0])
    try:
        pts = io.load_txt(path.replace(f"/{ftree[-1]}", "_points.tsv"))
    except FileNotFoundError:
        pt = (0, 0, 0)
    else:
        try:
            pt = pts[pidx, -3:]
        except IndexError:
            pt = pts[-3:]
    return LightPointKD(scn, parent=parent, src=ftree[-2], posidx=pidx, pt=pt)


stypes = ('1comp', '2comp', '3comp', '1compdv', 'directview', 'directpatch', 'sunonly',
          'sunpatch', 'skyonly')

stypedescriptions = {
    '1comp': "standard DC method, sky patch only, full contribution depending "
             "on skyengine settings",
    '2comp': "sky patch for sky contribution, sun run for sun contribution, "
             "depth of contributions depends on skyengine and sunengine "
             "settings, no approximation for sun from sky patch",
    '3comp': "2-phase DDS, sky handles sky+indirect sun, sun handles direct sun"
             " requires directskyrun -ab 1 and sunrun -ab 0",
    '1compdv': "standad DC method, but with direct view replacement of sun and"
               " specular reflections",
    'directview': "only evaluate srcviewpts (direct views to sun and specular "
                  "reflections",
    'directpatch': "only evaluate results from dskyrun",
    'sunonly': "only evaluate results from sunrun",
    'sunpatch': "use skyrun results to evaluate sun contribution",
    'skyonly': "use skyrun to evaluate sky contribution only"
    }

stypedocstring = "\n".join([f"    - {k}: {v}" for k,v in stypedescriptions.items()])


def get_integrator(scn, pm, srcname="suns", simtype="2comp",
                   sunviewengine=None):
    req_sun = ('2comp', '3comp', 'directview', 'sunonly')
    req_sky = ('1comp', '2comp', '3comp', 'sunpatch', 'skyonly', '1compdv')
    req_dsk = ('3comp', 'directpatch', '1compdv')
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
                   includesun=simtype != "skyonly", sunviewengine=sunviewengine)
    if simtype == "2comp":
        return Integrator(skyplane, sunplane, sunviewengine=sunviewengine)
    if simtype == "3comp":
        return Integrator(skyplane, dskplane, sunplane,
                          sunviewengine=sunviewengine, ds=True)
    if simtype == "1compdv":
        return Integrator(skyplane, dskplane, sunviewengine=sunviewengine,
                          dv=True)
    if simtype in ["directview", "sunonly"]:
        return Integrator(sunplane, includesky=False,
                          sunviewengine=sunviewengine)
    if simtype == "directpatch":
        return Integrator(dskplane, includesky=False,
                          sunviewengine=sunviewengine)
    try:
        srcpoints = f"{scn.outdir}/{pm.name}/{simtype}_points.tsv"
        srcplane = LightPlaneKD(scn, srcpoints, pm, simtype)
    except OSError:
        raise ValueError(f"Error loading {simtype}")
    else:
        return Integrator(srcplane, sunviewengine=sunviewengine)
