# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

# code adapted from supplemental materials to:
# Andrew B. Watson; A formula for human retinal ganglion cell receptive
# field density as a function of visual field location.
# Journal of Vision 2014;14(7):15. doi: https://doi.org/10.1167/14.7.15.
# Measured data (referenced in  RetinalTopography.nb from above comes from:
# Curcio, C. A., Sloan, K. R., Kalina, R. E., & Hendrickson, A. E. (1990).
# Human photoreceptor topography. J Comp Neurol, 292 (4), 497 - 523
# and
# Curcio, C. A., & Allen, K. A. (1990). Topography of ganglion cells in
# human retina. The Journal of comparative neurology, 300 (1), 5 - 25

import numpy as np


def rgcf_density_on_meridian(deg, mi):
    """retinal ganlgion cell field density along a meridian as a functional
    best fit.

    the field density accounts for the input region of the ganglion cell to
    account for displaced ganglion cells. This value is estimate from cone
    density and the inferred density of midget ganglion cells. see Watson (2014)
    for important caveats.

    Parameters
    ----------
    deg: np.array
        eccentricity in degrees along merdian
    mi: int
        meridian index. [0, 1, 2, 3] for Temporal, Superior, Nasal, Inferior.

    Returns
    -------
    np.array
        1d array of retinal ganglion cell density along a merdian

    """
    foveal_rgc_density = 33162.4
    # meridian_rgcf_density coefficients
    mrdc = np.array((
        (0.9851255243509307, 1.057910790284875, 22.139309670912258),
        (0.9934935604656688, 1.0354962329628912, 16.346430181031927),
        (0.9729319961934142, 1.0841663616483523, 7.632575316432455),
        (0.9960391452747578, 0.993222146189384, 12.131840717293096)))
    mrd = mrdc[mi]
    return (foveal_rgc_density * mrd[:, 0] * np.power(1 + deg / mrd[:, 1], -2) +
            (1 - mrd[:, 0]) * np.exp(-deg / mrd[:, 2]))


def rgc_density_on_meridian(deg, mi):
    """retinal ganglion cell density along a merdian as a linear interpolation
    between non-zero measurements

    As opposed to the field density this estimate the actual location of
    ganglion cells, which could be important to consider for intrinsically
    photosensitive cells. These are (partially?) responsible for pupillary
    response. However, even iprgc (may?) receive signals from rods/cones

    Parameters
    ----------
    deg: np.array
        eccentricity in degrees along merdian
    mi: int
        meridian index. [0, 1, 2, 3] for Temporal, Superior, Nasal, Inferior.

    Returns
    -------
    np.array
        1d array of retinal ganglion cell density along a merdian
    """
    meridians = [np.array(_curcio_nozero[0]), np.array(_curcio_nozero[1]),
                 np.array(_curcio_nozero[2]), np.array(_curcio_nozero[3])]
    interp = []
    for m in meridians:
        interp.append(np.interp(deg, m[:, 0], m[:, 1]))
    mrd = np.stack(interp)[mi, np.arange(len(deg))]
    return np.maximum(mrd, 1)


def rgcf_density_xy(xy, func=rgcf_density_on_meridian):
    """interpolate density between meridia, selected by quadrant

    Parameters
    ----------
    xy: np.array
        xy visual field coordinates on a disk in degrees
        (eccentricity 0-90 from fovea)
    func: callable
        density function along a meridian, takes r in degrees and an axes index:
        [0, 1, 2, 3] for Temporal, Superior, Nasal, Inferior.

    Returns
    -------
    np.array
        1d array of single eye densities
    """
    r = np.linalg.norm(xy, axis=-1)
    xmi = np.where(xy[:, 0] > 0, 0, 2)
    ymi = np.where(xy[:, 0] > 0, 1, 3)
    dx = func(r, xmi)
    dy = func(r, ymi)
    xy2 = np.square(xy)
    xy2[:, 0] = xy2[:, 0] / dx
    xy2[:, 1] = xy2[:, 1] / dy
    den = np.sum(xy2, axis=-1)
    return np.where(r == 0, dy, np.square(r) / den)


def binocular_density(xy, func=rgcf_density_on_meridian):
    """average denisty between both eyes.

    Parameters
    ----------
    xy: np.array
        xy visual field coordinates on a disk (eccentricity 0-1 from fovea)
    func: callable
        density function along a meridian, takes r in degrees and an axes index:
        [0, 1, 2, 3] for Temporal, Superior, Nasal, Inferior. coordinates are
        for the visual field.

    Returns
    -------
    np.array
        1d array of average binocular densities
    """
    xyd = 90*xy
    d1 = rgcf_density_xy(xyd, func)
    xyd[:, 0] *= -1
    d2 = rgcf_density_xy(xyd, func)
    return (d1 + d2)/2


def rgcf_density(xy):
    """retinal ganglion cell field density

    Parameters
    ----------
    xy: np.array
        xy visual field coordinates on a disk (eccentricity 0-1 from fovea)

    Returns
    -------
    np.array
        1d array retinal ganglion cell field density according to
        model by Watson
    """
    return binocular_density(xy)


def rgc_density(xy):
    """retinal ganglion cell density (includes displaced ganglion cells)

    Parameters
    ----------
    xy: np.array
        xy visual field coordinates on a disk (eccentricity 0-1 from fovea)

    Returns
    -------
    np.array
        1d array retinal ganglion cell density according to
        measurements by Curcio

    """
    return binocular_density(xy, rgc_density_on_meridian)


# coefficients copied from RetinalTopography.nb, with zeros
# (normal incidence and blindspots) removed
_curcio_nozero = (((0.18593923369252877, 44.04486724637118),
                   (0.3717178308822736, 93.27509002277617),
                   (0.5573316409090776, 198.78698645599684),
                   (0.7427764977750474, 345.14206545230155),
                   (1.1131426529212247, 1016.4921674895295),
                   (1.4827829593647686, 1460.4333317652136),
                   (1.8516644783823826, 1828.1252513740205),
                   (2.2197552940730496, 1960.6584606225276),
                   (2.5870255432599225, 2085.475117966171),
                   (2.953449149712472, 2217.2767820329545),
                   (3.31900679290571, 2342.4640796660046),
                   (3.6836910241589105, 2374.660330589903),
                   (5.496092431124066, 2069.255695654906),
                   (7.309190492700514, 1282.5353001727826),
                   (9.142240526078217, 751.4476414029601),
                   (10.994765968171452, 492.43006927557184),
                   (18.522503804692484, 244.45459692983894),
                   (22.312720415066067, 226.03104078393105),
                   (26.097857900564726, 209.0325120219425),
                   (29.870308487985934, 181.4036385525174),
                   (33.62997699705354, 143.7728154334466),
                   (37.38428195373141, 110.33967350723363),
                   (41.148156075669945, 90.3683973832319),
                   (44.944046636790006, 79.71178905772537),
                   (48.80191581130344, 69.9320213451311),
                   (52.75924101405819, 61.47100679480568),
                   (56.86101523023512, 53.48768790671183),
                   (61.159747316488954, 46.62792067725425),
                   (65.71546224905477, 38.43392514810544),
                   (70.5957012915439, 30.498345341427296),
                   (75.87552205717101, 23.54151243161036),
                   (81.63749844776936, 15.280986323342862),
                   (87.97172046481518, 10.732960056617229)
                   ),
                  ((0.550233963180544, 28.588082376804778),
                   (0.7335011377041551, 119.37582656017271),
                   (1.0998704064881881, 700.1324809546082),
                   (1.4660819859706975, 1375.1623704831711),
                   (1.8322149031328192, 1848.0020695754924),
                   (2.1983507190480474, 2044.233823113446),
                   (2.564570904219949, 2092.6703340121226),
                   (2.9309542225042526, 2098.755365542933),
                   (3.2975743653080882, 2112.379709683535),
                   (3.664498030757884, 2039.0232001188388),
                   (5.505390523026586, 1387.2230758663704),
                   (7.358901171835787, 871.9981716115124),
                   (9.224896045436125, 590.3718199737535),
                   (11.101263681414402, 431.9496814466633),
                   (14.874340393583799, 241.0785460949621),
                   (18.658178224708905, 156.73195466309647),
                   (22.43835579663088, 114.03852703528533),
                   (26.20747855963523, 97.00162933290599),
                   (29.965510328268024, 75.5295020408025),
                   (33.7198675919376, 60.035746714586786),
                   (37.48545310663173, 43.94237419037801),
                   (41.28467077707197, 31.26095732464047),
                   (45.14743387650181, 26.46259676627039),
                   (49.111170564943365, 23.901916201454696),
                   (53.220828095818824, 21.387873950004174),
                   (57.52887615397506, 18.80405892440746),
                   (62.095309375899646, 15.789364331254303),
                   (66.98764894026573, 12.10655110734033),
                   (72.28094307870904, 8.770651414360177),
                   (78.05776640401032, 7.352182662490541),
                   (84.40821806124717, 5.3828977637443405)
                   ),
                  ((0.18609574024132236, 13.642224855792355),
                   (0.3723438821360879, 49.018225489997356),
                   (0.5587403493942099, 119.42980837318217),
                   (0.7452810972349723, 303.3539776112752),
                   (1.1187794270092395, 820.236588824208),
                   (1.492807224442807, 1250.5519036556952),
                   (1.8673334627883122, 1593.832907000932),
                   (2.2423277698263666, 1692.332668322065),
                   (2.617760452197751, 1806.4929444368368),
                   (2.9936025129379713, 1926.5199197102822),
                   (3.369825664284743, 2012.1877464547917),
                   (3.7464023371508843, 2015.0019767880601),
                   (5.6336746771345725, 1707.4759437387086),
                   (7.526071665400324, 1210.7695431481452),
                   (9.420978792024693, 833.677620814168),
                   (11.316249083753902, 586.3104294462349),
                   (15.101635111623818, 283.2349068144286),
                   (18.87443425652101, 165.84758925059035),
                   (22.63436212527552, 115.59487769209959),
                   (26.388647406362324, 76.9537932292463),
                   (30.152032737336533, 55.009907495087226),
                   (33.94677525485396, 41.8629871004271),
                   (37.80264703601664, 30.820864683843475),
                   (41.75693549104231, 24.99016095054558),
                   (45.85444371717299, 20.110113240656887),
                   (50.147490801803194, 16.840968392033965),
                   (54.695912052189826, 12.931766264533087),
                   (59.567059126021746, 10.026405148168433),
                   (64.83580004103732, 8.62649206015602)
                   ),
                  ((0.1835809385909668, 37.391520240434666),
                   (0.3672580363229842, 90.12699185434538),
                   (0.5510372355174309, 249.437121068217),
                   (0.7349238911274849, 493.2108300406277),
                   (1.1030380830755004, 1082.453292457224),
                   (1.471631994701548, 1610.2758922129483),
                   (1.8407284168030345, 1983.3089650782533),
                   (2.2103423139296576, 2031.5422102177708),
                   (2.5804817064036194, 2069.678912807843),
                   (2.951148586786319, 2105.454751383192),
                   (3.322339812544252, 2100.948307894678),
                   (3.6940479379285027, 1988.5925574830246),
                   (5.559867472191449, 1348.2086272974666),
                   (7.436093405865805, 727.5067813469979),
                   (9.320030408557752, 490.5855896552269),
                   (11.209002125908963, 312.92986055035055),
                   (14.992807030840048, 144.1646454133318),
                   (18.77304101986144, 94.74293408347762),
                   (22.542253120449665, 71.35548046728823),
                   (26.300346373716106, 53.40502365532766),
                   (30.05467462322393, 41.78054273431498),
                   (33.82007722367664, 33.1849696825539),
                   (37.6188944785438, 26.219971064184996),
                   (41.48097615682201, 21.84238246440211),
                   (45.443687172993215, 18.88391803873515),
                   (49.55191186952844, 17.81137621351253),
                   (53.85805736281627, 16.654949662512383),
                   (58.42205600935188, 14.234065864687464),
                   (63.31136688553662, 10.222039436761069),
                   (68.60097614179438, 5.971059086995101),
                   (74.37339614738148, 2.8085175111507112),
                   (80.7186634582328, 0.6778900972815473)
                   )
                  )
