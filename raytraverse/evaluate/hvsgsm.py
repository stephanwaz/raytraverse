# -*- coding: utf-8 -*-
# Copyright (c) 2022 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import warnings

import numpy as np
from scipy.ndimage import convolve, gaussian_filter

import clasp.script_tools as cst
from raytraverse import io
from raytraverse.utility import imagetools as itl
from raytraverse.utility import pool_call
from raytraverse.mapper import ViewMapper


def gss_compute(imgs, illums=None, save=False, suffix="_rg.hdr", outdir=None,
                **kwargs):
    """initialize a GSS instance and compute multiple images in parallel

    Parameters
    ----------
    imgs: Sequence
        list of image file paths to compute. images should by 180 degree HDR
        angular fisheyes scaled at 1/179 cd/m^2 (standard radiance HDR)
    illums: Sequence, optional
        If images onnly contain glare sources but not an accurate background
        provide illuminance calculated seperately (like eDGPs process)
    save: bool, optional
        If true saves an image of the glare response
    suffix: str, optional
        suffix to append to image when save is True
    outdir: str, optional
        save response images to a different directory
    kwargs:
        passed to GSS initialization

    Returns
    -------
    GSS: list
        glare sensation scores for all images (in order given)
    """
    if outdir is not None and save:
        cst.try_mkdir(outdir)
    gss = GSS(imgs[0], **kwargs)
    if illums is not None:
        imgs = list(zip(imgs, illums))
    else:
        imgs = list(zip(imgs, [None]*len(imgs)))
    return pool_call(process_gss, imgs, gss, outdir=outdir, outf=save,
                     suffix=suffix, desc="calculating GSS", expandarg=True)


def process_gss(img, illum, ins, outf=False, outdir=None, suffix="_rg.hdr"):
    """called by gss_compute in parallel"""
    ins.lum = img
    if outf:
        saverg = img.rsplit(".", 1)[0] + suffix
        if outdir is not None:
            saverg = outdir + "/" + saverg.rsplit("/")[-1]
    else:
        saverg = None
    return ins.compute(save=saverg, ev_eye=illum)


def f_b(b, c, phi):
    """component of point spread function

    J.K. Ijspeert, T.J.T.P. Van Den Berg, H. Spekreijse, An improved
        mathematical description of the foveal visual point spread function
        with parameters for age, pupil size and pigmentation,
        Vision Research, Volume 33, Issue 1, 1993,Pages 15-20, ISSN 0042-6989,
        https://doi.org/10.1016/0042-6989(93)90053-Y.
    """
    return c*b/(2*np.pi*np.power(np.square(np.sin(phi)) +
                                 b**2*np.square(np.cos(phi)), 1.5))


def l_b(b, c, phi):
    """component of line spread function

        J.K. Ijspeert, T.J.T.P. Van Den Berg, H. Spekreijse, An improved
            mathematical description of the foveal visual point spread function
            with parameters for age, pupil size and pigmentation,
            Vision Research, Volume 33, Issue 1, 1993,Pages 15-20, ISSN 0042-6989,
            https://doi.org/10.1016/0042-6989(93)90053-Y.
        """
    return c*b/(np.pi*(np.square(np.sin(phi)) + b**2*np.square(np.cos(phi))))


class GSS:
    """calculate GSS for images with angular fisheye projection

    application of model described in:

        A GENERIC GLARE SENSATION MODEL BASED ON THE HUMAN VISUAL SYSTEM
        Vissenberg, M.C.J.M., Perz, M., Donners, M.A.H., Sekulovski, D.
        Signify Research, Eindhoven, THE NETHERLANDS
        gilles.vissenberg@signify.com
        DOI 10.25039/x48.2021.0P23

    see methods for citations associated with each step in model.

    the model requires the following steps:

    Done when setting an image with a new resolution:
    1. calculate solid angle of pixels
    2. calculate eccentricity from guth position idx

    Steps for applying model to an image:
    1. calculate eye illuminance from image
    2. mask non-glare source pixels (not described in model, fixed thresh)
    3. calculate pupil area and diameter
    4. calculate global retinal irradiance
    5. calculate incident retinal irradiance of glare sources
    6. apply PSF to (5)
    7. apply movement affecting adaptation to (6)
    8. apply movement affecting direct response to (6)
    9. calculate local adaptation using (7)
    10. calculate V/V_m photoreceptor response (8)
    11. calculate receptor field response to (10) as DoG
    12. normalize field response with logistic
    13. apply position weighting
    14. sum GSS

    Parameters
    ----------
    view:
        can be None, a view file, a ViewMapperr, or an hdrimage with
        a valid view specification (must be -vta)
    gst:
        glare source threshold (cd/m^2)
    age:
        age of observer
    f:
        eye focal length
    scale:
        factor to apply to raw pixel values to convert to cd/m^2
    pigmentation:
        from Ijspeert et al. 1993:
            mean for blue eyes: 0.16
            brown eyes: 0.106
            dark brown eyes: 0.056
    fwidth: Union[int, float], optional
        the width of the frame for psf
    psf: bool, optional
        apply pointspread function for light arriving at retina
    adaptmove: bool, optional
        apply involuntary eye movement effect on local adaptation
    directmove: book, optional
        apply involuntary eye movement effect on direct cone response
    raw: bool, optional
        do not weight results, used for calibration

    Notes
    -----
    set self.lum, either by initializing with an image, or with the
    parameter setter, then compute::

        gss = GSS("img.hdr")
        gss.lum = "img.hdr"
        score = gss.compute()

    additional images can be loaded and computed with the parameter setter
    by calling images with the same resolution and view size on an
    initialized object, subsantial re-computation can be avoided.

    Alternatively, to get access to process arrays or to override pupil
    adaptation and or isolating glare sources::

        e_g, pupa, pupd = self.adapt(ev_eye)
        img_gs = self.get_glare_sources()
        r_g, parrays = self.glare_response(img_gs, e_g, pupa, pupd,
        return_arrays=True)

    For processing multiple images with the same GSS initialization in
    parallel, see hvsgsm.gss_compute()
    """

    # eccentricity min and max scaling for field width in response
    # from Vissenberg et al. (.12, 0.009)
    emax = 0.12
    emin = 0.009

    # coefficients for logistic response function
    # from Vissenberg et al. (22, 0.25)
    fr_a = 22
    fr_b = 0.25

    # coeficient for DoG in linear response
    # from Vissenberg et al.  0.67
    fr_k = 0.67

    # minkowski normalization exponent
    # from Vissenberg et al. 4
    norm = 4

    # contrast for eye adaptation (higher value, less global adaptation:
    # 0.8 indoor (medium)
    # 0.9 outdoor (high) -- maybe daylight is always here?
    contrast = 0.8

    def __init__(self, view=None, gst=0, age=40, f=16.67, scale=179,
                 pigmentation=0.106, fwidth=10, psf=True, adaptmove=True,
                 directmove=True, raw=False):
        self.gst = gst
        self.age = age
        self.f = f
        self.scale = scale
        self.pigmentation = pigmentation
        self.fwidth = fwidth
        self._blur = (psf, adaptmove, directmove)
        self._raw = raw
        # initialize properties set when an image is loaded
        self._res = 0
        self._vecs = None
        self._omega = None
        self._mask = None
        self._nmask = None
        self._lum = None
        self._sigma_c = None
        self._pparcmin = None
        if view is None:
            self.vm = ViewMapper(viewangle=180)
        elif isinstance(view, ViewMapper):
            self.vm = view
        else:
            try:
                self.vm = itl.hdr2vm(view)
            except ValueError:
                self.vm = ViewMapper(viewangle=180)
            except IOError:
                self.vm = itl.vf_to_vm(view)
            else:
                self.lum = view

    def adapt(self, ev_eye=None):
        """step 1 in compute, adapt eye to image"""
        if self.lum is None:
            raise ValueError("cannot adapt until an image has been set")
        if ev_eye is None:
            ev_eye = np.einsum('i,i,i->', self.ctheta.flat[self.mask],
                               self.lum.flat[self.mask],
                               self.omega.flat[self.mask])
        pupa, pupd = self.pupil(ev_eye)
        e_g = self.retinal_irradiance(ev_eye/np.pi, pupa)
        return e_g, pupa, pupd

    def get_glare_sources(self):
        """step 2 in compute, isolate glare sources"""
        if self.lum is None:
            raise ValueError("cannot get_glare_sources until an image has been "
                             "set")
        img_gs = np.copy(self.lum, 'C')
        img_gs[self.lum < self.gst] = 0
        return img_gs

    def glare_response(self, img_gs, e_g, pupa, pupd, return_arrays=False):
        """step 3 in compute, apply steps of Vissenberg et al. model

        Parameters
        ----------
        img_gs: np.array
            representing all glare sources
        e_g: float
            global retinal irradiance
        pupa: float
            pupil area (mm^2)
        pupd: float
            pupil diameter (mm)
        return_arrays: bool, optional
            if True returns second value with dict of process arrays
            else return r_w only

        Returns
        -------
        r_w: np.array
            weighted glare response for entire retina as represented by image
        parrays: dict, optional
            with returned_arrays=True
            keys: retinal_irrad, psf, adapt_eye_movement, direct_eye_movement,
            local_adaptation, response_ratio, response_lin, response_log
        """
        if self.lum is None:
            raise ValueError("cannot glare_response until an image has been set")
        parrays = dict()
        e_r = self.retinal_irradiance(img_gs, pupa)
        parrays["01_retinal_irrad"] = e_r
        if self._blur[0]:
            e_r_psf = self.apply_psf(e_r, pupd)
            parrays["02_psf"] = e_r_psf
        else:
            e_r_psf = e_r
        if self._blur[1]:
            e_rg = self.apply_eye_movement_1(e_r_psf)
            parrays["03_adapt_eye_movement"] = e_rg
        else:
            e_rg = e_r_psf
        if self._blur[2]:
            e_rc = self.apply_eye_movement_2(e_r_psf, e_g)
            parrays["04_direct_eye_movement"] = e_rc
        else:
            e_rc = e_r_psf
        e_a = self.local_eye_adaptation(e_rg, e_g)
        parrays["05_local_adaptation"] = e_a
        vvm = self.cone_response(e_rc, e_a)
        parrays["06_response_ratio"] = vvm
        # r_lin
        r_rf = self.field_response(vvm)
        parrays["07_response_lin"] = r_rf
        r_g = self.normalized_field_response(r_rf)
        bfill = np.zeros(r_g.shape[1:])
        parrays["08_response_log"] = np.stack((*r_g, bfill))
        if self._raw:
            r_w = r_g
        else:
            r_w = self.weight_response(r_g)
            parrays["09_weighted_response"] = np.stack((*r_g, bfill))
        if return_arrays:
            return r_w, parrays
        else:
            return r_w

    def compute(self, save=None, ev_eye=None):
        """apply glare sensation model to loaded image

        Parameters
        ----------
        save: str
            if given save response image to file specified (.hdr)
        ev_eye: float, opttional
            externally calculated Ev

        Returns
        -------
        float
        """
        if self.lum is None:
            raise ValueError("cannot compute until an image has been set")
        e_g, pupa, pupd = self.adapt(ev_eye)
        img_gs = self.get_glare_sources()
        r_g = self.glare_response(img_gs, e_g, pupa, pupd)
        if save is not None:
            r_gc = np.stack((*r_g, np.zeros(r_g.shape[1:])))
            io.carray2hdr(r_gc, save)
        return self.gss(r_g)

    @property
    def lum(self):
        return self._lum

    @lum.setter
    def lum(self, img):
        """reads image and updates view based values if necessary"""
        self._lum = io.hdr2array(img).T*self.scale
        r = self._lum.shape[0]
        if self._res != r:
            self._vecs = self.vm.pixelrays(r)
            self._omega = self.vm.pixel2omega(self.vm.pixels(r), r)
            self._mask = self.vm.in_view(self._vecs, False)
            self._nmask = np.logical_not(self._mask)
            self._omega.flat[self._nmask] = 0
            ct = np.maximum(0, self.vm.ctheta(self._vecs))
            self._ctheta = ct.reshape(self._lum.shape)
            self._res = r
            self._pparcmin = self.res/(60*self.vm.viewangle)
            self.sigma_c = r

    @property
    def res(self):
        """resolution, set via lum"""
        return self._res

    @property
    def vecs(self):
        """directions, set via lum"""
        return self._vecs

    @property
    def omega(self):
        """solid angle, set via lum"""
        return self._omega

    @property
    def mask(self):
        """view mask, set via lum"""
        return self._mask

    @property
    def ctheta(self):
        """cos between vectors and view direction, set via lum"""
        return self._ctheta

    @property
    def sigma_c(self):
        """position index scaled to eccentricity .009-.12
        (used in field_response)

        Note that this differs from the implementation dscribed by Vissenberg
        et al., and uses ganglion cell field density from:

        Andrew B. Watson; A formula for human retinal ganglion cell receptive
        field density as a function of visual field location. Journal of
        Vision 2014;14(7):15. doi: https://doi.org/10.1167/14.7.15.
        """
        return self._sigma_c

    @sigma_c.setter
    def sigma_c(self, r):
        # # model described by vissenberg with greater acceptance below horizon
        xy = self.vm.xyz2vxy(self._vecs.reshape(-1, 3))*2 - 1
        vecc = np.where(xy[:, 1] >= 0, xy[:, 1]/0.61111, xy[:, 1]/0.94444)
        p = np.minimum(np.sqrt(np.square(xy[:, 0]/1.05556) +
                       np.square(vecc)), 1)
        s_c = p*(self.emax - self.emin) + self.emin
        self._sigma_c = s_c.reshape(r, r)

    @property
    def vm(self):
        return self._vm

    @vm.setter
    def vm(self, view):
        if isinstance(view, ViewMapper):
            vm = view
        else:
            try:
                vm = itl.hdr2vm(view)
            except ValueError:
                vm = None
            if vm is None:
                vm = ViewMapper(viewangle=180)
        self._vm = vm

    # Step 3
    def pupil(self, ev):
        """calculate pupil area

        Based on:
        Donners, Maurice & Vissenberg, Michel & Geerdinck, L.M. & Broek-Cools,
        J. (2015). A PSYCHOPHYSICAL MODEL OF DISCOMFORT GLARE IN BOTH OUTDOOR
        AND INDOOR APPLICATIONS.

        Parameters
        ----------
        ev:
            illumiance at eye (lux)
        """
        pd = (5 - 3*np.tanh(0.4*np.log10(ev/1600)) -
              0.043*(self.age - 10)*np.exp(-ev/174))
        return np.pi*(pd/2)**2, pd

    # Step 4
    def retinal_irradiance(self, lum, pupa):
        """adjust incident light on retina based on pupil size and focal-length

        from Vissenberg et al. 2021 equation (1):
        (1) E_r = A_p * L / f^2
        E_r: local retinal irrradiance
        L: field luminance
        """
        return (pupa / self.f**2) * lum

    def prep_kernel(self):
        """construct an array to hold a kernel scaled to image resolution
        """
        # approximates that relationship of pixels in cone is constant
        window = int(np.ceil(self.fwidth*60*self._pparcmin))
        vm = ViewMapper(viewangle=self.fwidth)
        img, vecs, mask, _, _ = vm.init_img(window)
        phi = vm.radians(vecs[mask])
        oga = vm.pixel2omega(self.vm.pixels(window), window)
        return img, phi, oga, mask

    def psf_coef(self, pupd):
        """age, pupil size and pigmentation adjusted PSF coefficients

        PSF:
            PSF(phi) = sum(c * f_b(phi))
            f_b(phi) = b/(2π * (sin^2(phi) + b^2*cos^2(phi))^1.5) 1/steradian

        LSF:
            LSF(phi) = sum(c * l_b(phi))
            l_b(phi) = b/(π * (sin^2(phi) + b^2*cos^2(phi))) 1/rad

        based on:
        J.K. Ijspeert, T.J.T.P. Van Den Berg, H. Spekreijse, An improved
        mathematical description of the foveal visual point spread function
        with parameters for age, pupil size and pigmentation,
        Vision Research, Volume 33, Issue 1, 1993,Pages 15-20, ISSN 0042-6989,
        https://doi.org/10.1016/0042-6989(93)90053-Y.
        """
        d = 3.2
        agefactor = 1 + np.power(self.age/70, 4)
        b = 9000 - 936 * np.sqrt(agefactor)
        e = np.sqrt(agefactor) / 2000
        pf = 1/self.pigmentation - 1
        csa = 1 / (1 + agefactor/pf)
        cla = 1 / (1 + pf/agefactor)
        pd = (1 + (pupd/d)**2)
        dp = (1 + (d/pupd)**2)
        cs = [csa/pd, csa/dp, cla/((1 + 25*self.pigmentation)*(1 + 1/agefactor))]
        cs += [cla - cs[-1]]
        bs = [pd / (b * pupd), dp * (e - 1/(b * pupd)),
              1 / (10 + 60 * self.pigmentation - 5 / agefactor), 1]
        return bs, cs

    # Step 6
    def apply_psf(self, e_r, pupd):
        """apply human foveal point spread function

        based on:
        J.K. Ijspeert, T.J.T.P. Van Den Berg, H. Spekreijse, An improved
        mathematical description of the foveal visual point spread function
        with parameters for age, pupil size and pigmentation,
        Vision Research, Volume 33, Issue 1, 1993,Pages 15-20, ISSN 0042-6989,
        https://doi.org/10.1016/0042-6989(93)90053-Y.
        """
        bs, cs = self.psf_coef(pupd)
        # calculate how much of PSF is in our window using high resolution
        # spacing on LSF
        lphi = np.linspace(-self.fwidth/2, self.fwidth/2,
                           int(self.fwidth/.002/2)*2 + 1)*np.pi/180
        dlphi = lphi[1] - lphi[0]
        lsf = np.sum([l_b(b, c, lphi) for b, c in zip(bs, cs)], axis=0)
        cap_eg = np.sum(lsf*dlphi)

        # approximate remainder with constant (the flat part)
        e_gs = np.einsum('i,i->', e_r.flat[self.mask],
                         self.omega.flat[self.mask])
        base = e_gs * (1 - cap_eg) / np.sum(self.omega.flat[self.mask])
        # build PSF filter
        psfk, phi, oga, mask = self.prep_kernel()
        psfk[mask] = np.sum([f_b(b, c, phi) for b, c in zip(bs, cs)], axis=0)
        # apply pixel solid angle
        psfk *= oga
        # hack to fix peak to normalize to 1 (needed because center PSF will
        # always be sharper than pixel resolution
        pk = np.unravel_index(psfk.argmax(), psfk.shape)
        psfk[pk] = 0
        psfk[pk] = cap_eg - np.sum(psfk)

        # apply filter than balance energy outside window
        e_r_psf = convolve(e_r, psfk)
        e_r_psf += base
        return e_r_psf

    # Step 7
    def apply_eye_movement_1(self, e_r):
        """eye movement gaussian adaptation model to blur image at the time-
        scale of adaptation response.

        based on:
        R. A. Normann, B. S. Baxter, H. Ravindra and P. J. Anderton,
        "Photoreceptor contributions to contrast sensitivity: Applications
        in radiological diagnosis," in IEEE Transactions on Systems, Man,
        and Cybernetics, vol. SMC-13, no. 5, pp. 944-953, Sept.-Oct. 1983,
        doi: 10.1109/TSMC.1983.6313090.

        Parameters
        ----------
        e_r: np.array
            retinal irradiance (optical correction)

        Returns
        -------
        adapt_eye_movement:
            retinal irradiance (with adaptation scale movement and optical
            correction)
        """
        # e ^ -1/2 (x/sigma)^2
        a = 0.62
        # B: 0.167
        # sigma = 1 / (sqrt(2) * B)
        bs = 4.2341723424 * self._pparcmin
        c = 0.38
        # D: .05
        # sigma = 1 / (sqrt(2) * D)
        ds = 14.1421356237 * self._pparcmin
        # Norman describes a truncation at 30 arcmin
        trunc = 30 * self._pparcmin / ds
        e_rg = (a * gaussian_filter(e_r, bs) +
                c * gaussian_filter(e_r, ds, truncate=trunc))
        # e_rg.flat[self._nmask] = 0
        return e_rg

    # Step 8
    def apply_eye_movement_2(self, e_r, e_g):
        """blur image due to eye movement during direct response

        from Vissenberg et al. 2021 equations (5) and (6):
        (5) τ = 100/(E_g * f^2)^0.12 ms
        tau (τ): cone integration time

        (6) w = 2 * sqrt(D * τ)
        D = 30.0 arcmin^2 * s^-1 (occular drift)
        D = 250.0 (micro saccades)

        Parameters
        ----------
        e_r: np.array
            retinal irradiance (optical correction)
        e_g: float
            global retinal irrradiance

        Returns
        -------
        direct_eye_movement:
            retinal irradiance (with movement and optical correction)
        """
        t = .1 / np.power(e_g * self.f**2, 0.12)
        # D = 30.0 arcmin^2 * s^-1 (occular drift)
        w1 = 2 * np.sqrt(30 * t) * self._pparcmin
        # D = 250.0 (micro saccades)
        w2 = 2 * np.sqrt(250 * t) * self._pparcmin
        e_ra = gaussian_filter(e_r, w1)
        e_ra = gaussian_filter(e_ra, w2)
        # e_ra.flat[self._nmask] = 0
        return e_ra

    # Step 9
    def local_eye_adaptation(self, e_r, e_g):
        """calculate locallized eye adaptation

        from Vissenberg et al. 2021 equation (4):
        log_10(E_a) = p * log_10(E_r) + (1-p) * log_10(E_g)
        E_a: adaptation illuminance
        p: 0.8 (indoor / moderate) - 0.9 (outdoor / strong) contrast

        Parameters
        ----------
        e_r: np.array
            retinal irradiance (optical correction)
        e_g: float
            global retinal irrradiance

        Returns
        -------
        local_adaptation
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log10er = np.where(e_r > 0, np.log10(e_r), 0)
        e_a = np.power(10, self.contrast * log10er +
                       (1-self.contrast) * np.log10(e_g))
        # e_a.flat[self._nmask] = 0
        return e_a

    # Step 10
    @staticmethod
    def cone_response(e_r, e_a):
        """calculate local response as a fraction of maximum at current
        adaptation

        from Vissenberg et al. 2021 equations (2) and (3):
        (2) V/V_m = E_r^n / (E_r^n + σ^n)
        V: photoreceptor response
        V_m: maximum response
        E_r: local retinal illuminance (apply w to this E_r)
        n: 0.74

        (3) σ = (5.701055^(1/2.55) + E_a^(1/2.55))^2.55
        sigma (σ): half-saturation retinal illuminance value

        Parameters
        ----------
        e_r: np.array
            retinal irradiance (with movement and optical correction)
        e_a: np.array
            local adaptation

        Returns
        -------
        response_ratio
        """
        n = 0.74
        sigma = np.power(5.701055**(1/2.55) + np.power(e_a, 1/2.55), 2.55)
        ern = np.power(e_r, n)
        vvm = ern/(ern + sigma**n)
        return vvm

    # Step 11
    def field_response(self, vvm):
        """receptive field response

        from Vissenberg et al. 2021 equation (7)::

            R_RF(r) = e^(-r^2/(2σ_c^2)) / (2πσ_c^2)
                      - K * e^(-r^2/(2σ_s^2)) / (2πσ_s^2)

        R_RF: receptive field response
        r: distance to receptive field center (degrees)
        σ_c: gaussian width of center (0.009 (center) - 0.12 (edge FOV) degrees)
        σ_s: gaussian width of surround 3.5 * σ_c
        K: DoG balance factor 0.67

        Parameters
        ----------
        vvm: np.array
            response_ratio (saturation)

        Returns
        -------
        response_lin
            linear, difference of gussians
        """
        rf_c = np.zeros(vvm.shape)
        rf_s = np.zeros(vvm.shape)
        # relative change in eccentricity is more important, so log is more
        # efficient
        steps = np.exp(np.linspace(np.log(self.emin), np.log(self.emax), 31))
        # steps = np.linspace(self.emin, self.emax, 16)
        ubounds = np.concatenate(((steps[1:] + steps[:-1])/2, [1]))
        lbounds = np.concatenate(([0], ubounds[:-1]))
        for s, lb, ub in zip(steps, lbounds, ubounds):
            sp = s * self._pparcmin*60
            r_c = gaussian_filter(vvm, sp)
            r_s = gaussian_filter(vvm, sp * 3.5)
            include = np.logical_and(lb <= self.sigma_c, self.sigma_c < ub)
            rf_c[include] = r_c[include]
            rf_s[include] = r_s[include]
        return rf_c - self.fr_k*rf_s

    # Step 12
    def normalized_field_response(self, r):
        """normalized non-linear ganglion response

        from Vissenberg et al. 2021 equation (8):
        R_G = 1 / (1 + e^(-a * (R_lin - b)))
        R_G: normalized non-linear ganglion response
        a: slope of logistic = 22
        b: 0.25

        Parameters
        ----------
        r: np.array
            response_lin

        Returns
        -------
        response_log
            logistic
        """
        r_gc = 1/(1 + np.exp(-self.fr_a*(r - self.fr_b)))
        # track as a seperate receptive field
        r_go = 1/(1 + np.exp(self.fr_a*(r + self.fr_b)))
        r_gc.flat[self._nmask] = 0
        r_go.flat[self._nmask] = 0
        return np.stack((r_gc, r_go))

    # Step 13
    def weight_response(self, r):
        """weight rectified response by position index

        Parameters
        ----------
        r: np.array
            response_log

        Returns
        -------
        position weighted glare response

        Notes
        -----
        fit on guth data using BCD = 2843.58 * e^(x + 1.5 * x^2) / 179
        with a 2.12 degree source and 34.26 cd/m^2 background

        numpy.polynomial.Polynomial.fit(x, y, 6)
        where x = eccentricity (.009 -.12 from 0 to 55 degree vertical angle
        and y = 1/unweighted GSS

        results::

            17.078747601175937 - 14.392547712049184·x¹ + 13.521269552690162·x² -
            8.778008624382208·x³ - 6.1589701503713865·x⁴ + 14.405349284130853·x⁵ -
            1.2184994327746506·x⁶ - 4.797592024869671·x⁷

        """
        p = np.polynomial.Polynomial([17.078747601175937, -14.392547712049184,
                                      13.521269552690162, -8.778008624382208,
                                      -6.1589701503713865, 14.405349284130853,
                                      -1.2184994327746506, -4.797592024869671],
                                     domain=[0.009, 0.12])
        return r * p(self.sigma_c)

    # Step 14
    def gss(self, r_g):
        """calculate minkowski sum on normalized response

        from Vissenberg et al. 2021 equation (9):
        (9) GSS = sum_i(R_G,i^m 𝛿_i)^(1/m)
        GSS: glare sensation score
        m: minkowski norm (4)
        delta (𝛿): solid angle of pixel (steradians)
        """
        return np.sum(np.power(r_g, self.norm) * self.omega)**(1/self.norm)





