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
 *  rcontribparts.c - partial modified rcontrib.c  (includes at top),
 *  splits call to rcontrib (as rcontrib_call)
 *  to return without exiting and to read input from a file instead of stdin.
 */

#ifdef __cplusplus
extern "C" {
#endif



/*
* Accumulate ray contributions for a set of materials
* Initialization and calculation routines
*/

#include "copyright.h"

#include "rcontrib.h"
#include "otypes.h"
#include "source.h"
#include "rcinit.h"

char	*shm_boundary = NULL;		/* boundary of shared memory */

CUBE	thescene;			/* our scene */
OBJECT	nsceneobjs;			/* number of objects in our scene */

int	dimlist[MAXDIM];		/* sampling dimensions */
int	ndims = 0;			/* number of sampling dimensions */
int	samplendx = 0;			/* index for this sample */

void	(*trace)() = NULL;		/* trace call (NULL before rcinit) */

int	do_irrad = 0;			/* compute irradiance? */

int	rand_samp = 1;			/* pure Monte Carlo sampling? */

double	dstrsrc = 0.9;			/* square source distribution */
double	shadthresh = 0.;		/* shadow threshold */
double	shadcert = .75;			/* shadow certainty */
int	directrelay = 3;		/* number of source relays */
int	vspretest = 512;		/* virtual source pretest density */
int	directvis = 1;			/* sources visible? */
double	srcsizerat = .2;		/* maximum ratio source size/dist. */

COLOR	cextinction = BLKCOLOR;		/* global extinction coefficient */
COLOR	salbedo = BLKCOLOR;		/* global scattering albedo */
double	seccg = 0.;			/* global scattering eccentricity */
double	ssampdist = 0.;			/* scatter sampling distance */

double	specthresh = .02;		/* specular sampling threshold */
double	specjitter = 1.;		/* specular sampling jitter */

int	backvis = 1;			/* back face visibility */

int	maxdepth = -10;			/* maximum recursion depth */
double	minweight = 2e-3;		/* minimum ray weight */

char	*ambfile = NULL;		/* ambient file name */
COLOR	ambval = BLKCOLOR;		/* ambient value */
int	ambvwt = 0;			/* initial weight for ambient value */
double	ambacc = 0.;			/* ambient accuracy */
int	ambres = 256;			/* ambient resolution */
int	ambdiv = 350;			/* ambient divisions */
int	ambssamp = 0;			/* ambient super-samples */
int	ambounce = 1;			/* ambient bounces */
char	*amblist[AMBLLEN+1];		/* ambient include/exclude list */
int	ambincl = -1;			/* include == 1, exclude == 0 */

int	account;			/* current accumulation count */
RNUMBER	raysleft;			/* number of rays left to trace */
long	waitflush;			/* how long until next flush */

RNUMBER	lastray = 0;			/* last ray number sent */
RNUMBER	lastdone = 0;			/* last ray output */

static void	trace_contrib(RAY *r);	/* our trace callback */

static void mcfree(void *p) { epfree((*(MODCONT *)p).binv); free(p); }

LUTAB	modconttab = LU_SINIT(NULL,mcfree);	/* modifier lookup table */

/************************** INITIALIZATION ROUTINES ***********************/

char *
formstr(				/* return format identifier */
        int  f
)
{
  switch (f) {
    case 'a': return("ascii");
    case 'f': return("float");
    case 'd': return("double");
    case 'c': return(COLRFMT);
  }
  return("unknown");
}


/* Add modifier to our list to track */
MODCONT *
addmodifier(char *modn, char *outf, char *prms, char *binv, int bincnt)
{
  LUENT	*lep = lu_find(&modconttab,modn);
  MODCONT	*mp;
  EPNODE	*ebinv;
  int	i;

  if (lep->data != NULL) {
    sprintf(errmsg, "duplicate modifier '%s'", modn);
    error(USER, errmsg);
  }
  if (nmods >= MAXMODLIST) {
    sprintf(errmsg, "too many modifiers (%d limit)", MAXMODLIST);
    error(INTERNAL, errmsg);
  }
  if (!strcmp(modn, VOIDID)) {
    sprintf(errmsg, "cannot track '%s' modifier", VOIDID);
    error(USER, errmsg);
  }
  modname[nmods++] = modn;	/* XXX assumes static string */
  lep->key = modn;		/* XXX assumes static string */
  if (binv == NULL)
    binv = "0";		/* use single bin if unspecified */
  ebinv = eparse(binv);
  if (ebinv->type == NUM) {	/* check value if constant */
    bincnt = (int)(evalue(ebinv) + 1.5);
    if (bincnt != 1) {
      sprintf(errmsg, "illegal non-zero constant for bin (%s)",
              binv);
      error(USER, errmsg);
    }
  } else if (bincnt <= 0) {
    sprintf(errmsg,
            "unspecified or illegal bin count for modifier '%s'",
            modn);
    error(USER, errmsg);
  }
  /* initialize results holder */
  mp = (MODCONT *)malloc(sizeof(MODCONT)+sizeof(DCOLOR)*(bincnt-1));
  if (mp == NULL)
    error(SYSTEM, "out of memory in addmodifier");
  mp->outspec = outf;		/* XXX assumes static string */
  mp->modname = modn;		/* XXX assumes static string */
  mp->params = prms;		/* XXX assumes static string */
  mp->binv = ebinv;
  mp->bin0 = 0;
  mp->nbins = bincnt;
  memset(mp->cbin, 0, sizeof(DCOLOR)*bincnt);
  /* figure out starting bin */
  while (!getostream(mp->outspec, mp->modname, mp->bin0, 1))
    mp->bin0++;
  /* allocate other output streams */
  for (i = 0; ++i < mp->nbins; )
    getostream(mp->outspec, mp->modname, mp->bin0+i, 1);
  lep->data = (char *)mp;
  return(mp);
}


/* Add modifiers from a file list */
void
addmodfile(char *fname, char *outf, char *prms, char *binv, int bincnt)
{
  char	*mname[MAXMODLIST];
  int	i;
  /* find the file & store strings */
  i = wordfile(mname, MAXMODLIST, getpath(fname, getrlibpath(), R_OK));
  if (i < 0) {
    sprintf(errmsg, "cannot find modifier file '%s'", fname);
    error(SYSTEM, errmsg);
  }
  if (i >= MAXMODLIST-1) {
    sprintf(errmsg, "too many modifiers (%d limit) in file '%s'",
            MAXMODLIST-1, fname);
    error(INTERNAL, errmsg);
  }
  for (i = 0; mname[i]; i++)	/* add each one */
    addmodifier(mname[i], outf, prms, binv, bincnt);
}


/* Check if we have any more rays left (and report progress) */
int
morays(void)
{
  static RNUMBER	total_rays;
  static time_t	tstart, last_report;
  time_t		tnow;

  if (!raysleft)
    return(1);	/* unknown total, so nothing to do or say */

  if (report_intvl > 0 && (tnow = time(0)) >= last_report+report_intvl) {
    if (!total_rays) {
      total_rays = raysleft;
      tstart = tnow;
    } else {
      sprintf(errmsg, "%.2f%% done after %.3f hours\n",
              100.-100.*raysleft/total_rays,
              (1./3600.)*(tnow - tstart));
      eputs(errmsg);
    }
    last_report = tnow;
  }
  return(--raysleft);
}


/************************** MAIN CALCULATION PROCESS ***********************/

/* Our trace call to sum contributions */
static void
trace_contrib(RAY *r)
{
  MODCONT	*mp;
  double	bval;
  int	bn;
  RREAL	contr[3];

  if (r->ro == NULL || r->ro->omod == OVOID)
    return;
  /* shadow ray not on source? */
  if (r->rsrc >= 0 && source[r->rsrc].so != r->ro)
    return;

  mp = (MODCONT *)lu_find(&modconttab,objptr(r->ro->omod)->oname)->data;

  if (mp == NULL)				/* not in our list? */
    return;

  worldfunc(RCCONTEXT, r);		/* else set context */
  set_eparams((char *)mp->params);
  if ((bval = evalue(mp->binv)) <= -.5)	/* and get bin number */
    return;				/* silently ignore negatives */
  if ((bn = (int)(bval + .5)) >= mp->nbins) {
    sprintf(errmsg, "bad bin number (%d ignored)", bn);
    error(WARNING, errmsg);
    return;
  }
  raycontrib(contr, r, PRIMARY);		/* compute coefficient */
  if (contrib)
    multcolor(contr, r->rcol);	/* -> contribution */
  addcolor(mp->cbin[bn], contr);
}


/* Evaluate irradiance contributions */
static void
eval_irrad(FVECT org, FVECT dir)
{
  RAY	thisray;

  VSUM(thisray.rorg, org, dir, 1.1e-4);
  thisray.rdir[0] = -dir[0];
  thisray.rdir[1] = -dir[1];
  thisray.rdir[2] = -dir[2];
  thisray.rmax = 0.0;
  rayorigin(&thisray, PRIMARY, NULL, NULL);
  /* pretend we hit surface */
  thisray.rxt = thisray.rot = 1e-5;
  thisray.rod = 1.0;
  VCOPY(thisray.ron, dir);
  VSUM(thisray.rop, org, dir, 1e-4);
  samplendx++;			/* compute result */
  (*ofun[Lamb.otype].funp)(&Lamb, &thisray);
}


/* Evaluate radiance contributions */
static void
eval_rad(FVECT org, FVECT dir, double dmax)
{
  RAY	thisray;
  /* set up ray */
  VCOPY(thisray.rorg, org);
  VCOPY(thisray.rdir, dir);
  thisray.rmax = dmax;
  rayorigin(&thisray, PRIMARY, NULL, NULL);
  samplendx++;			/* call ray evaluation */
  rayvalue(&thisray);
}


/* Accumulate and/or output ray contributions (child or only process) */
static void
done_contrib(void)
{
  MODCONT	*mp;
  int	i;

  if (account <= 0 || --account)
    return;			/* not time yet */

  for (i = 0; i < nmods; i++) {	/* output records & clear */
    mp = (MODCONT *)lu_find(&modconttab,modname[i])->data;
    mod_output(mp);
    memset(mp->cbin, 0, sizeof(DCOLOR)*mp->nbins);
  }
  end_record();			/* end lines & flush if time */

  account = accumulate;		/* reset accumulation counter */
}



int
getvecfp(FVECT vec, FILE *fp)
{
  float	vf[3];
  double	vd[3];
  char	buf[32];
  int	i;

  switch (inpfmt) {
    case 'a':					/* ascii */
      for (i = 0; i < 3; i++) {
        if (fgetword(buf, sizeof(buf), fp) == NULL ||
            !isflt(buf))
          return(-1);
        vec[i] = atof(buf);
      }
      break;
    case 'f':					/* binary float */
      if (getbinary((char *)vf, sizeof(float), 3, fp) != 3)
        return(-1);
      VCOPY(vec, vf);
      break;
    case 'd':					/* binary double */
      if (getbinary((char *)vd, sizeof(double), 3, fp) != 3)
        return(-1);
      VCOPY(vec, vd);
      break;
    default:
      error(CONSISTENCY, "botched input format");
  }
  return(0);
}

/************************** MODIFIED PARTS ***********************/

FILE*
rcinit2(char *fname)
{
  int	i;

  if (nproc > MAXPROCESS)
    sprintf(errmsg, "too many processes requested -- reducing to %d",
            nproc = MAXPROCESS);
  if (nproc > 1) {
    preload_objs();		/* preload auxiliary data */
    /* set shared memory boundary */
    shm_boundary = strcpy((char *)malloc(16), "SHM_BOUNDARY");
  }
  trace = trace_contrib;		/* set up trace call-back */
  for (i = 0; i < nsources; i++)	/* tracing to sources as well */
    source[i].sflags |= SFOLLOW;
  if (yres > 0) {			/* set up flushing & ray counts */
    if (xres > 0)
      raysleft = (RNUMBER)xres*yres;
    else
      raysleft = yres;
  } else
    raysleft = 0;
  if ((account = accumulate) > 1)
    raysleft *= accumulate;
  waitflush = (yres > 0) & (xres > 1) ? 0 : xres;
  if (nproc > 1 && in_rchild2())	/* forked child? */{
    return stdin;      /* return to main processing loop */
  }

  if (recover) {			/* recover previous output? */
    if (accumulate <= 0)
      reload_output();
    else
      recover_output();
  }
  if (nproc == 1)	{ /* single process? */
    FILE *fp;
    if (fname == NULL)
      fp = stdin;
    else if ((fp = fopen(fname, "rb")) == NULL) {
      fprintf(stderr, "cannot open input file \"%s\"", fname);
      error(SYSTEM, errmsg);
    }
    return fp;
  }
  /* else run appropriate controller */
  if (accumulate <= 0) {
    feeder_loop2(fname);
  }
  else {
    parental_loop2(fname);
  }
  return NULL;
}

/* Quit program */
void
quitrc(
        int  code
)
{
  if (nchild > 0)		/* close children if any */
    end_children(code != 0);
  else if (nchild < 0)
    _exit(code);	/* avoid flush() in child */
  exit(code);
}


void rcontrib_call(char *fname){
  static int	ignore_warning_given = 0;
  FVECT		orig, direc;
  double		d;
  FILE *fp;
  fp = rcinit2(fname);
  if (fp == NULL){
    return;
  }
  else if (fp != stdin) {
  #ifdef getc_unlocked
    flockfile(fp);
  #endif
  }
  while (getvecfp(orig, fp) == 0 && getvecfp(direc, fp) == 0) {
    d = normalize(direc);
    if (nchild != -1 && (d == 0.0) & (accumulate == 0)) {
      if (!ignore_warning_given++)
        error(WARNING,
              "dummy ray(s) ignored during accumulation\n");
      continue;
    }
    if (lastray+1 < lastray)
      lastray = lastdone = 0;
    ++lastray;
    if (d == 0.0) {				/* zero ==> flush */
      if ((yres <= 0) | (xres <= 1))
        waitflush = 1;		/* flush after */
      if (nchild == -1)
        account = 1;
    } else if (imm_irrad) {			/* else compute */
      eval_irrad(orig, direc);
    } else {
      eval_rad(orig, direc, lim_dist ? d : 0.0);
    }
    done_contrib();		/* accumulate/output */
    ++lastdone;
    if (raysleft && !--raysleft)
      break;		/* preemptive EOI */
  }

  if (nchild != -1 && (accumulate <= 0) | (account < accumulate)) {
    if (account < accumulate) {
      error(WARNING, "partial accumulation in final record");
      accumulate -= account;
    }
    account = 1;		/* output accumulated totals */
    done_contrib();
  }
  if (fp != stdin) {
#ifdef getc_unlocked
    funlockfile(fp);    /* avoid lock/unlock overhead */
#endif
    fclose(fp);
  }
  if (raysleft)
    error(USER, "unexpected EOF on input");
  quitrc(0);
}

void rcontrib_clear(){
//  lu_doall(&modconttab, &lu_delete, NULL);
  for (int i = 0; i < nmods; i++)
    lu_delete(&modconttab, modname[i]);
  nmods = 0;
}



#ifdef __cplusplus
}
#endif
