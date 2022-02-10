# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""functions for translating between coordinate spaces and resolutions"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter, uniform_filter
from scipy.spatial import cKDTree


def norm(v):
    """normalize 2D array of vectors along last dimension"""
    return v/np.linalg.norm(v, axis=-1).reshape(-1, 1)


def norm1(v):
    """normalize flat vector"""
    n = np.sqrt(np.sum(np.square(v)))
    if n == 0:
        n = 1
    return np.asarray(v)/n


###############################################
# Shirley=Chiu Disk to Square Transformations #
###############################################

def uv2xy(uv):
    """translate from unit square (0,1),(0,1) to disk (x,y)
    http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric
    -map.html.
    """
    uv = np.atleast_2d(uv)
    np.seterr(all="ignore")
    xy = np.empty((uv.shape[0], 2), float)
    u = uv[:, 0]
    v = uv[:, 1]
    n = np.where(u > 1.0, -1, 1)
    u = np.where(u > 1.0, u - 1.0, u)
    a = 2. * u - 1
    b = 2. * v - 1
    cond = a*a > b*b
    r = np.where(cond, a, np.where(b == 0, 0, b))
    phi = np.where(cond, b/(2*a), np.where(b == 0, 0, 1 - a/(2*b)))*np.pi/2
    xy[:, 0] = n*np.cos(phi)*r
    xy[:, 1] = np.sin(phi)*r
    return xy


def uv2xyz(uv, axes=(0, 1, 2), xsign=1):
    """translate from 2 x unit square (0,2),(0,1) to unit sphere (x,y,z)
    http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric
    -map.html.
    """
    np.seterr(all="ignore")
    uv = np.atleast_2d(uv)
    xyz = np.empty((uv.shape[0], 3), float)
    u = uv[:, 0]
    v = uv[:, 1]
    # u > 1 values lay in the negative z hemisphere
    n = np.where(u > 1.0, -1, 1)
    # bring both hemispheres to (0,1)
    u = np.where(u > 1.0, u - 1.0, u)
    a = 2. * u - 1
    b = 2. * v - 1
    cond = a*a > b*b
    r = np.where(cond, a, np.where(b == 0, 0, b))
    phi = np.where(cond, b/(2*a), np.where(b == 0, 0, 1 - a/(2*b)))*np.pi/2
    sphterm = r*np.sqrt(2 - r*r)
    # flip back x in positive z space
    xyz[:, axes[0]] = xsign*n*np.cos(phi)*sphterm
    xyz[:, axes[1]] = np.sin(phi)*sphterm
    # add sign to z
    xyz[:, axes[2]] = n*(1 - r*r)
    return xyz


def xyz2uv(xyz, normalize=False, axes=(0, 1, 2), flipu=False):
    """translate from vector x,y,z (normalized) to u,v (0,2),(0,1)
    Shirley, Peter, and Kenneth Chiu. A Low Distortion Map Between Disk and
    Square. Journal of Graphics Tools, vol. 2, no. 3, Jan. 1997, pp. 45-52.
    Taylor and Francis+NEJM, doi:10.1080/10867651.1997.10487479.
    """
    xyz = np.atleast_2d(xyz)
    if normalize:
        xyz = norm(xyz)
    uv = np.empty((xyz.shape[0], 2), float)
    # store sign of z-axis to map both hemispheres as positive
    n = np.where(xyz[:, axes[2]] < 0, -1, 1)
    r2 = 1 - n*xyz[:, axes[2]]
    x = xyz[:, axes[0]] / np.sqrt(2 - r2)
    y = xyz[:, axes[1]] / np.sqrt(2 - r2)
    r = np.sqrt(np.square(x) + np.square(y))
    phi = np.arctan2(y, x)
    pi4 = np.pi/4
    phi = phi + np.where(phi < -pi4, 2*np.pi, 0)
    a = np.where(phi < pi4, (r, phi*r/pi4),
                 np.where(phi < 3*pi4, (-(phi - np.pi/2)*r/pi4, r),
                          np.where(phi < 5*pi4, (-r, -(phi - np.pi)*r/pi4),
                                   ((phi - 3*np.pi/2)*r/pi4, -r)))).T
    # for the positive z-direction (n=1) map -1,1 to 1,0 (to correct flip)
    # for the negative z-direction (n=-1) map -1,1 to 1,2
    if flipu:
        uv[:, 0] = (np.where(n < 0, 3, 1) - a[:, 0]*n) / 2.
    else:
        uv[:, 0] = (a[:, 0] + 2 - n) / 2.
    uv[:, 1] = (a[:, 1] + 1) / 2.
    return uv


###########################################
# Translate to/from shirley chiu sky bins #
###########################################

scbinscal = ("""
{ map U/V axis to bin divisions }
axis(x) : mod(floor(side * x), side);
nrbins = side * side;
{ get bin of u,v }
binl(u, v) : axis(u)*side + axis(v);

{ shirley-chiu disk to square (with spherical term) }
pi4 : PI/4;
n = if(Dz, 1, -1);
r2 = 1 - n*Dz;
x = Dx/sqrt(2 - r2);
y = -Dy/sqrt(2 - r2);
r = sqrt( sq(x) + sq(y));
ph = atan2(x, y);
phi = ph + if(-pi4 - ph, 2*PI, 0);
a = if(pi4 - phi, r, if(3*pi4 - phi, -(phi - PI/2)*r/pi4, if(5*pi4 - phi,"""
             """ -r, (phi - 3*PI/2)*r/pi4)));
b = if(pi4 - phi, phi*r/pi4, if(3*pi4 - phi, r, if(5*pi4 - phi, """
             """-(phi - PI)*r/pi4, -r)));

{ map to (0,2),(0,1) matches raytraverse.translate.xyz2uv}
U = (if(n, 1, 3) - a*n)/2;
V = (b + 1)/2;

bin = if(n, binl(V, U), nrbins);

{ for visualizing with gridlines }
t : .015;
grid(x) : if(and(inside(t, frac(U*side), 1-t), inside(t, frac(V*side), 1-t)), x, 0);
""")

scxyzcal = """
x1 = .5;
x2 = .5;

U = ((bin - mod(bin, side)) / side + x1)/side;
V = (mod(bin, side) + x2)/side;

n = if(U - 1, -1, 1);
ur = if(U - 1, U - 1, U);
a = 2 * ur - 1;
b = 2 * V - 1;
conda = sq(a) - sq(b);
condb = abs(b) - FTINY;
r = if(conda, a, if(condb, b, 0));
phi = if(conda, b/(2*a), if(condb, 1 - a/(2*b), 0)) * PI/2;
sphterm = r * sqrt(2 - sq(r));
Dx = n * cos(phi)*sphterm;
Dy = sin(phi)*sphterm;
Dz = n * (1 - sq(r));
"""


def xyz2skybin(xyz, side, tol=0, normalize=False):
    xyz = np.atleast_2d(xyz)
    uv = xyz2uv(xyz, flipu=False, normalize=normalize)
    if tol > 0:
        tol = tol/side
        uvi = np.linspace(-tol, tol, 3)
        uvs = np.stack(np.meshgrid(uvi, uvi)).reshape(2, 9).T + uv
        sbin = np.unique(uv2bin(uvs, side)).astype(int)
        skybin = sbin[sbin < side**2]
    else:
        skybin = uv2bin(uv, side)
    return skybin


def skybin2xyz(bn, side):
    """generate source vectors from sky bins

    Parameters
    ----------
    bn: np.array
        bin numbers
    side: int
        square side of discretization

    Returns
    -------
    xyz: np.array
        direction to center of sky patches
    """
    uv = bin2uv(bn, side)
    xyz = uv2xyz(uv, xsign=1)
    xyz[xyz[:, 2] < 0] = (0, 0, -1)
    return xyz


##################################################
# Translate to/from anglular fisheye projeection #
##################################################

def xyz2xy(xyz, axes=(0, 1, 2), flip=False):
    """xyz coordinates to xy mapping of angular fisheye proejection"""
    xyz = np.atleast_2d(xyz)
    r = np.arctan2(np.sqrt(np.sum(np.square(xyz[:, axes[0:2]]), -1)),
                   xyz[:, axes[2]])/(np.pi/2)
    phi = np.arctan2(xyz[:, axes[0]], xyz[:, axes[1]])
    x = r * np.sin(phi)
    y = r * np.cos(phi)
    if flip:
        x = -x
    return np.stack((x, y)).T


##########################################
# Translate to/from anglular coordinates #
##########################################

def tpnorm(thetaphi):
    """normalize angular vector to 0-pi, 0-2pi"""
    thetaphi = np.atleast_2d(thetaphi)
    thetaphi[:, 0] = np.mod(thetaphi[:, 0] + np.pi, np.pi)
    thetaphi[:, 1] = np.mod(thetaphi[:, 1] + 2*np.pi, 2*np.pi)
    return thetaphi


def tp2xyz(thetaphi, normalize=True):
    """calculate x,y,z vector from theta (0-pi) and phi (0-2pi) RHS Z-up"""
    thetaphi = np.atleast_2d(thetaphi)
    if normalize:
        thetaphi = tpnorm(thetaphi)
    theta = thetaphi[:, 0]
    phi = thetaphi[:, 1]
    sint = np.sin(theta)
    xyz = np.array([sint*np.cos(phi), sint*np.sin(phi),
                    np.cos(theta)]).T
    return norm(xyz)


def xyz2tp(xyz):
    """calculate theta (0-pi), phi from x,y,z RHS Z-up"""
    xyz = np.atleast_2d(xyz)
    theta = np.arccos(xyz[:, 2])
    phi = np.where(np.isclose(theta, 0.0, atol=1e-10), np.pi,
                   np.where(np.isclose(theta, np.pi, atol=1e-10),
                            np.pi, np.arctan2(xyz[:, 1], xyz[:, 0])))
    return tpnorm(np.column_stack([theta, phi]))


def tp2uv(thetaphi):
    """calculate UV from theta (0-pi), phi"""
    thetaphi = np.atleast_2d(thetaphi)
    return xyz2uv(tp2xyz(thetaphi))


def uv2tp(uv):
    """calculate theta (0-pi), phi from UV"""
    uv = np.atleast_2d(uv)
    return xyz2tp(uv2xyz(uv))


def aa2xyz(aa):
    """calculate altitude (0-90), azimuth (-180,180) from xyz"""
    aa = np.atleast_2d(aa)
    tp = np.pi/2 - aa * np.pi/180
    tp[:, 1] += np.pi
    return tp2xyz(tp)


def xyz2aa(xyz):
    """calculate xyz from altitude (0-90), azimuth (-180,180)"""
    xyz = np.atleast_2d(xyz)
    tp = xyz2tp(xyz)
    tp[:, 1] -= np.pi
    return (np.pi/2 - tp)/(np.pi/180)


def chord2theta(c):
    """compute angle from chord on unit circle

    Parameters
    ----------
    c: float
        chord or euclidean distance between normalized direction vectors

    Returns
    -------
    theta: float
        angle captured by chord
    """
    return 2*np.arcsin(c/2)


def theta2chord(theta):
    """compute chord length on unit sphere from angle

    Parameters
    ----------
    theta: float
        angle

    Returns
    -------
    c: float
        chord or euclidean distance between normalized direction vectors
    """
    return 2*np.sin(theta/2)


def ctheta(a, b):
    """cos(theta) (dot product) between a and b"""
    a = np.asarray(a)
    b = np.asarray(b)
    return np.clip(np.einsum("i,ji->j", a, b.reshape(-1, b.shape[-1])), -1, 1)


def radians(a, b):
    """angle in radians betweeen a and b"""
    return np.arccos(ctheta(a, b))


def degrees(a, b):
    """angle in degrees  betweeen a and b"""
    return radians(a, b) * 180/np.pi

################################################
# digitize and serialize UV square coordinates #
################################################

def uv2ij(uv, side, aspect=2):
    uv = np.atleast_2d(uv)
    ij = np.mod(np.floor(side*uv), side)
    if aspect == 2:
        ij[:, 0] += (uv[:, 0] >= 1) * side
    return ij.astype(int)


def uv2bin(uv, side):
    buv = uv2ij(uv, side)
    return buv[:, 0]*side + buv[:, 1]


def bin2uv(bn, side, offset=0.5):
    u = ((bn - np.mod(bn, side))/side + offset)/side
    v = (np.mod(bn, side) + offset)/side
    return np.stack((u, v)).T


#########################
# image like resampling #
#########################

def resample(samps, ts=None, gauss=True, radius=None):
    """simple array resampling. requires whole number multiple scaling.

    Parameters
    ----------
    samps: np.array
        array to resample along each axis
    ts: tuple, optional
        shape of output array, should be multiple of samps.shape
    gauss: bool, optional
        apply gaussian filter to upsampling
    radius: float, optional
        when gauss is True, filter radius, default is the scale ratio - 1

    Returns
    -------
    np.array
        to resampled array

    """
    if ts is None:
        ts = samps.shape
    rs = np.array(ts)/np.array(samps.shape)
    if np.prod(rs) > 1:
        for i in range(len(rs)):
            samps = np.repeat(samps, rs[i], i)
        if gauss:
            if radius is None:
                radius = tuple(rs - 1)
            samps = gaussian_filter(samps, radius)
    elif np.prod(rs) < 1:
        rs = (1/rs).astype(int)
        og = (-rs/2).astype(int)
        samps = uniform_filter(samps, rs, origin=og)
        for i, j in enumerate(rs):
            samps = np.take(samps, np.arange(0, samps.shape[i], j), i)
    elif radius is not None:
        if gauss:
            samps = gaussian_filter(samps, radius)
        else:
            samps = uniform_filter(samps, int(radius))
    return samps


####################
# vector rotations #
####################

def rmtx_elem(theta, axis=2, degrees=True):
    if degrees:
        theta = theta * np.pi / 180
    rmtx = np.array([(np.cos(theta), -np.sin(theta), 0),
                     (np.sin(theta), np.cos(theta), 0),
                     (0, 0, 1)])
    return np.roll(rmtx, axis-2, (0, 1))


def rotate_elem(v, theta, axis=2, degrees=True):
    rmtx = rmtx_elem(theta, axis=axis, degrees=degrees)
    return np.einsum('ij,kj->ki', rmtx, v)


def rmtx_yp(v):
    """generate a pair of rotation matrices to transform from vector v to
    z, enforcing a z-up in the source space and a y-up in the destination. If
    v is z, returns pair of identity matrices, if v is -z returns pair of 180
    degree rotation matrices.

    Parameters
    ----------
    v: array-like of size (N, 3)
        the vector direction representing the starting coordinate space

    Returns
    -------

    ymtx, pmtx: (np.array, np.array)
        two rotation matrices to be premultiplied in order to reverse transform,
        swap order and transpose.

    Notes
    -----
    if N is one:
    Forward: (pmtx@(ymtx@xyz.T)).T or np.einsum("ij,kj,li->kl", ymtx, xyz, pmtx)
    Backward: (ymtx.T@(pmtx.T@xyz.T)).T or np.einsum("ji,kj,il-vkl", pmtx, nv, ymtx)
    else:
    Forward: np.einsum("vij,vkj,vli->vkl", ymtx, xyz, pmtx)
    Backward: np.einsum("vji,vkj,vil-vkl", pmtx, nv, ymtx)
    """

    vs = norm(np.asarray(v).reshape(-1, 3))
    v2 = np.array((0, 0, 1)).reshape(-1, 3)
    tp = xyz2tp(vs)

    # check for identity or reverse transforms
    e = np.all(np.isclose(vs, v2), 1)
    ne = np.all(np.isclose(vs, -v2), 1)

    z = np.zeros(len(vs))
    o = np.ones(len(vs))

    # yaw matrix
    y = 3*np.pi/2 - tp[:, 1]
    cy = np.cos(y)
    sy = np.sin(y)
    ymtx = np.stack(((cy, sy, z), (-sy, cy, z), (z, z, o))).T
    ymtx[e] = np.eye(3)
    ymtx[ne] = np.array([(-1, 0, 0), (0, -1, 0), (0, 0, 1)])

    # pitch matrix
    p = -tp[:, 0]
    cp = np.cos(p)
    sp = np.sin(p)
    pmtx = np.stack(((o, z, z), (z, cp, sp), (z, -sp, cp))).T
    pmtx[e] = np.eye(3)
    pmtx[ne] = np.array([(1, 0, 0), (0, -1, 0), (0, 0, -1)])

    return np.squeeze(ymtx), np.squeeze(pmtx)


def cull_vectors(vecs, tol):
    """return mask to cull duplicate vectors within tolerance

    Parameters
    ----------
    vecs: Union[cKDTree, np.array]
        prebuilt KDTree or np.array to build a new one. culling keeps
        first vector in array used to build tree.
    tol: float
        tolerance for culling

    Returns
    -------
    np.array
        boolean mask of vecs (or vecs.data) to cull vectors
    """
    if type(vecs) != cKDTree:
        pkd = cKDTree(vecs)
    else:
        pkd = vecs
        vecs = pkd.data
    pairs = pkd.query_ball_tree(pkd, tol)
    flt = np.full(len(vecs), True)
    # keep track of culled indices to avoid removing chains of points
    flagged = set()
    for j, p in enumerate(pairs):
        if j not in flagged and len(p) > 1:
            # don't purge first or current indices
            q = [i for i in p[1:] if i != j]
            flt[q] = False
            flagged.update(q)
    return flt
