/* Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
 * =======================================================================
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *=======================================================================
 *
 * The Following code is copied from, adapts, includes, and/or links parts of
 * the Radiance source code which is licensed by the following:
 *
 * The Radiance Software License, Version 1.0
 *
 * Copyright (c) 1990 - 2018 The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory.   All rights reserved.
 *
 * If a copy of the full text of the License was not distributed with this file
 * (in ./ray/License.txt) the License is available at
 * https://www.radiance-online.org/download-install/license
 */

/*
 *  rtraceparts.c - partial modified rtrace.c (includes at top), splits call to rtrace (as rtrace_call)
 *  to return without exiting.
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 *  rtrace.c - program and variables for individual ray tracing.
 */

#include "copyright.h"

/*
 *  Input is in the form:
 *
 *	xorg	yorg	zorg	xdir	ydir	zdir
 *
 *  The direction need not be normalized.  Output is flexible.
 *  If the direction vector is (0,0,0), then the output is flushed.
 *  All values default to ascii representation of real
 *  numbers.  Binary representations can be selected
 *  with '-ff' for float or '-fd' for double.  By default,
 *  radiance is computed.  The '-i' or '-I' options indicate that
 *  irradiance values are desired.
 */

#include  <time.h>

#include  "platform.h"
#include  "ray.h"
#include  "ambient.h"
#include  "source.h"
#include  "otypes.h"
#include  "otspecial.h"
#include  "random.h"

extern int repeat;  /* RAYTRAVERSE MODIFICATION number of times to repeat ray */
int return_value_count = 1;
extern char  *outvals;			/* output values */

extern int  imm_irrad;			/* compute immediate irradiance? */
extern int  lim_dist;			/* limit distance? */

extern char  *tralist[];		/* list of modifers to trace (or no) */
extern int  traincl;			/* include == 1, exclude == 0 */

extern int  hresolu;			/* horizontal resolution */
extern int  vresolu;			/* vertical resolution */

long putcount;
RREAL *output_values;

int  castonly = 0;			/* only doing ray-casting? */

#ifndef  MAXTSET
#define	 MAXTSET	8191		/* maximum number in trace set */
#endif
OBJECT	traset[MAXTSET+1]={0};		/* trace include/exclude set */

static RAY  thisray;			/* for our convenience */

static FILE  *inpfp = NULL;		/* input stream pointer */

typedef void putf_t(RREAL *v, int n);
putf_t putn;

typedef void oputf_t(RAY *r);
static oputf_t  oputo, oputd, oputv, oputV, oputl, oputL, oputc, oputp,
        oputr, oputR, oputx, oputX, oputn, oputN, oputs,
        oputw, oputW, oputm, oputM, oputtilde;

static void raycast(RAY *r);
static void rayirrad(RAY *r);
static void rtcompute(FVECT org, FVECT dir, double dmax);
static int printvals(RAY *r);

static oputf_t *ray_out[32], *every_out[32];
static putf_t *putreal;

static void
raycast(			/* compute first ray intersection only */
        RAY *r
)
{
  if (!localhit(r, &thescene)) {
    if (r->ro == &Aftplane) {	/* clipped */
      r->ro = NULL;
      r->rot = FHUGE;
    } else
      sourcehit(r);
  }
}

static void
rayirrad(			/* compute irradiance rather than radiance */
        RAY *r
)
{
  void	(*old_revf)(RAY *) = r->revf;
  /* pretend we hit surface */
  r->rxt = r->rot = 1e-5;
  VSUM(r->rop, r->rorg, r->rdir, r->rot);
  r->ron[0] = -r->rdir[0];
  r->ron[1] = -r->rdir[1];
  r->ron[2] = -r->rdir[2];
  r->rod = 1.0;
  /* compute result */
  r->revf = raytrace;
  (*ofun[Lamb.otype].funp)(&Lamb, r);
  r->revf = old_revf;
}


static void
rtcompute(			/* compute and print ray value(s) */
        FVECT  org,
        FVECT  dir,
        double	dmax
)
{
  /* set up ray */
  rayorigin(&thisray, PRIMARY, NULL, NULL);
  if (imm_irrad) {
    VSUM(thisray.rorg, org, dir, 1.1e-4);
    thisray.rdir[0] = -dir[0];
    thisray.rdir[1] = -dir[1];
    thisray.rdir[2] = -dir[2];
    thisray.rmax = 0.0;
    thisray.revf = rayirrad;
  } else {
    VCOPY(thisray.rorg, org);
    VCOPY(thisray.rdir, dir);
    thisray.rmax = dmax;
    if (castonly)
      thisray.revf = raycast;
  }
  if (ray_pnprocs > 1) {		/* multiprocessing FIFO? */
    if (ray_fifo_in(&thisray) < 0)
      error(USER, "lost children");
    return;
  }
  samplendx++;			/* else do it ourselves */
  rayvalue(&thisray);
  printvals(&thisray);
}

int printcount = 0;
COLOR accumulated_color = {0, 0, 0};

static int
printvals(			/* print requested ray values */
        RAY  *r
)
{
  oputf_t **tp;
  double sf = 1/(double)repeat;
  if (ray_out[0] == NULL)
    return(0);
  printcount = (printcount + 1) % repeat;
  addcolor(accumulated_color, r->rcol);
  if (printcount == 0) {
    scalecolor(accumulated_color, sf);
    copycolor(r->rcol, accumulated_color);
    scalecolor(accumulated_color, 0.0);
    for (tp = ray_out; *tp != NULL; tp++)
      (**tp)(r);
  }
  return(1);
}


static void
oputo(				/* print origin */
        RAY  *r
)
{
  (*putreal)(r->rorg, 3);
}


static void
oputd(				/* print direction */
        RAY  *r
)
{
  (*putreal)(r->rdir, 3);
}


static void
oputr(				/* print mirrored contribution */
        RAY  *r
)
{
  RREAL	cval[3];

  cval[0] = colval(r->mcol,RED);
  cval[1] = colval(r->mcol,GRN);
  cval[2] = colval(r->mcol,BLU);
  (*putreal)(cval, 3);
}



static void
oputR(				/* print mirrored distance */
        RAY  *r
)
{
  (*putreal)(&r->rmt, 1);
}


static void
oputx(				/* print unmirrored contribution */
        RAY  *r
)
{
  RREAL	cval[3];

  cval[0] = colval(r->rcol,RED) - colval(r->mcol,RED);
  cval[1] = colval(r->rcol,GRN) - colval(r->mcol,GRN);
  cval[2] = colval(r->rcol,BLU) - colval(r->mcol,BLU);
  (*putreal)(cval, 3);
}


static void
oputX(				/* print unmirrored distance */
        RAY  *r
)
{
  (*putreal)(&r->rxt, 1);
}


static void
oputv(				/* print value */
        RAY  *r
)
{
  RREAL	cval[3];

  cval[0] = colval(r->rcol,RED);
  cval[1] = colval(r->rcol,GRN);
  cval[2] = colval(r->rcol,BLU);
  (*putreal)(cval, 3);
}


static void
oputV(				/* print value contribution */
        RAY *r
)
{
  RREAL	contr[3];

  raycontrib(contr, r, PRIMARY);
  multcolor(contr, r->rcol);
  (*putreal)(contr, 3);
}


static void
oputl(				/* print effective distance */
        RAY  *r
)
{
  RREAL	d = raydistance(r);

  (*putreal)(&d, 1);
}


static void
oputL(				/* print single ray length */
        RAY  *r
)
{
  (*putreal)(&r->rot, 1);
}


static void
oputc(				/* print local coordinates */
        RAY  *r
)
{
  (*putreal)(r->uv, 2);
}


static RREAL	vdummy[3] = {0.0, 0.0, 0.0};
static RREAL	vdummy1[1] = {0.0};

static void
oputp(				/* print point */
        RAY  *r
)
{
  if (r->rot < FHUGE*.99)
    (*putreal)(r->rop, 3);
  else
    (*putreal)(vdummy, 3);
}


static void
oputN(				/* print unperturbed normal */
        RAY  *r
)
{
  if (r->rot < FHUGE*.99) {
    if (r->rflips & 1) {	/* undo any flippin' flips */
      FVECT	unrm;
      unrm[0] = -r->ron[0];
      unrm[1] = -r->ron[1];
      unrm[2] = -r->ron[2];
      (*putreal)(unrm, 3);
    } else
      (*putreal)(r->ron, 3);
  } else
    (*putreal)(vdummy, 3);
}


static void
oputn(				/* print perturbed normal */
        RAY  *r
)
{
  FVECT  pnorm;

  if (r->rot >= FHUGE*.99) {
    (*putreal)(vdummy, 3);
    return;
  }
  raynormal(pnorm, r);
  (*putreal)(pnorm, 3);
}


static void
oputs(				/* print name */
        RAY  *r
)
{
  if (r->ro != NULL)
    fputs(r->ro->oname, stdout);
  else
    putchar('*');
  putchar('\t');
}


static void
oputw(				/* print weight */
        RAY  *r
)
{
  RREAL	rwt = r->rweight;

  (*putreal)(&rwt, 1);
}


static void
oputW(				/* print coefficient */
        RAY  *r
)
{
  RREAL	contr[3];
  /* shadow ray not on source? */
  if (r->rsrc >= 0 && source[r->rsrc].so != r->ro)
    setcolor(contr, 0.0, 0.0, 0.0);
  else
    raycontrib(contr, r, PRIMARY);

  (*putreal)(contr, 3);
}


static void
oputm(				/* print modifier */
        RAY  *r
)
{
  if (r->ro != NULL)
    if (r->ro->omod != OVOID)
      fputs(objptr(r->ro->omod)->oname, stdout);
    else
      fputs(VOIDID, stdout);
  else
    putchar('*');
  putchar('\t');
}


static void
oputM(				/* print material */
        RAY  *r
)
{
  RREAL omod[1];
  omod[0] = 0.0;

  if (r->ro != NULL)
    omod[0] = r->ro->omod;
  (*putreal)(omod, 1);


//  OBJREC	*mat;
//  if (r->ro != NULL) {
//    if ((mat = findmaterial(r->ro)) != NULL)
//      fputs(mat->oname, stdout);
//    else
//      fputs(VOIDID, stdout);
//  } else
//    putchar('*');
//  putchar('\t');
}

void putn(RREAL *v, int n){ /* output to buffer */
  for (int i = 0; i < n; i++){
    output_values[putcount + i] = v[i];
  }
  putcount += n;
}

extern char	*shm_boundary;		/* boundary of shared memory */

extern void
rtrace_setup( /* initialize processes */
	int  nproc
)
{
  setambient();
  if (castonly || every_out[0] != NULL)
    nproc = 1;		/* don't bother multiprocessing */
  if (nproc > 1) {		/* start multiprocessing */
    ray_popen(nproc);
    ray_fifo_out = printvals;
  }
}

extern RREAL*
rtrace_call( /* run rtrace process */
        const double *vptr,
        int nproc,
        int raycount
)
{
  output_values = (RREAL *)malloc(sizeof(RREAL) * raycount * return_value_count);
  putcount = 0;
  int i = 0;
  rtrace_setup(nproc);
  double	d;
  int ti;
  FVECT  orig, direc;

  /* process input rays */
  while (raycount > 0) {
    raycount--;
    ti = i;
    i += 6;
    orig[0] = vptr[ti];
    orig[1] = vptr[ti+1];
    orig[2] = vptr[ti+2];
    direc[0] = vptr[ti+3];
    direc[1] = vptr[ti+4];
    direc[2] = vptr[ti+5];
    for (int r = 0; r < repeat; r++) {
      rtcompute(orig, direc, lim_dist ? d : 0.0);
    }
  }
  if (ray_pnprocs > 1) {				/* clean up children */
    if (ray_fifo_flush() < 0)
      error(USER, "unable to complete processing");
    ray_pclose(0);			/* close child processes */

    if (shm_boundary != NULL) {	/* clear shared memory boundary */
      free((void *)shm_boundary);
      shm_boundary = NULL;
    }
  }
  ambdone();
  return output_values;
}

void
oputrad(				/* print value -o spec: Z*/
        RAY  *r
)
{
  RREAL	lum = bright(r->rcol);
  (*putreal)(&lum, 1);
}

int
setoutput2(char *vs)      /* provides additional outputspec Z to output radiance (1 component)*/
{
  oputf_t **table = ray_out;
  int  ncomp = 0;
  return_value_count = 0;
  if (!*vs) {
    sprintf(errmsg, "empty output specification");
    goto badopt;
  }
  sprintf(errmsg, "empty output specification");
  castonly = 1;			/* sets castonly as side-effect */
  do
    switch (*vs) {
      case 'o':				/* origin */
        *table++ = oputo;
        ncomp += 3;
        break;
      case 'd':				/* direction */
        *table++ = oputd;
        ncomp += 3;
        break;
      case 'r':				/* reflected contrib. */
        *table++ = oputr;
        ncomp += 3;
        castonly = 0;
        break;
      case 'R':				/* reflected distance */
        *table++ = oputR;
        ncomp++;
        castonly = 0;
        break;
      case 'x':				/* xmit contrib. */
        *table++ = oputx;
        ncomp += 3;
        castonly = 0;
        break;
      case 'X':				/* xmit distance */
        *table++ = oputX;
        ncomp++;
        castonly = 0;
        break;
      case 'v':				/* value */
        *table++ = oputv;
        ncomp += 3;
        castonly = 0;
        break;
      case 'V':				/* contribution */
        *table++ = oputV;
        ncomp += 3;
        castonly = 0;
      case 'l':				/* effective distance */
        *table++ = oputl;
        ncomp++;
        castonly = 0;
        break;
      case 'c':				/* local coordinates */
        *table++ = oputc;
        ncomp += 2;
        break;
      case 'L':				/* single ray length */
        *table++ = oputL;
        ncomp++;
        break;
      case 'p':				/* point */
        *table++ = oputp;
        ncomp += 3;
        break;
      case 'n':				/* perturbed normal */
        *table++ = oputn;
        ncomp += 3;
        castonly = 0;
        break;
      case 'N':				/* unperturbed normal */
        *table++ = oputN;
        ncomp += 3;
        break;
      case 'M':				/* material */
        *table++ = oputM;
        ncomp++;
        break;
      case 'w':				/* weight */
        *table++ = oputw;
        ncomp++;
        break;
      case 'W':				/* coefficient */
        *table++ = oputW;
        ncomp += 3;
        castonly = 0;
        break;
      case 'Z':     /* photopic brightness value */
        *table++ = oputrad;
        castonly = 0;
        ncomp++;
        break;
      default:
        sprintf(errmsg, "unrecognized output option '%c'", *vs);
        goto badopt;
    }
  while (*++vs);

  *table = NULL;
  if (*every_out != NULL)
    ncomp = 0;
  /* compatibility */
  if ((do_irrad | imm_irrad) && castonly){
    sprintf(errmsg, "-I+ and -i+ options require some value output");
    goto badopt;
  }

  for (table = ray_out; *table != NULL; table++) {
    if ((*table == oputV) | (*table == oputW)){
      sprintf(errmsg, "-oVW options require trace mode");
      goto badopt;
    }
    if ((do_irrad | imm_irrad) &&
        (*table == oputr) | (*table == oputR) |
        (*table == oputx) | (*table == oputX)){
      sprintf(errmsg, "-orRxX options incompatible with -I+ and -i+");
      goto badopt;
    }

  }
  putreal = putn;
  return_value_count = ncomp;
  badopt:
  if (return_value_count < 1)
    return -1;
  return ncomp;
}



#ifdef __cplusplus
}
#endif
