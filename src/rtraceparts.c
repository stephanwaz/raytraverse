/* Copyright (c) 2020 Stephen Wasilewski
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

#include "rtrace.c"
#include "rtinit.h"

extern char	*shm_boundary;		/* boundary of shared memory */

extern void
rtrace_setup( /* initialize processes */
	int  nproc
)
{
  setambient();
  long  nextflush = (!vresolu | (hresolu <= 1)) * hresolu;
  if (castonly || every_out[0] != NULL)
    nproc = 1;		/* don't bother multiprocessing */
  if ((nextflush > 0) & (nproc > nextflush)) {
    error(WARNING, "reducing number of processes to match flush interval");
    nproc = nextflush;
  }
  if (nproc > 1) {		/* start multiprocessing */
    ray_popen(nproc);
    ray_fifo_out = printvals;
  }
}

extern void
rtrace_call( /* run rtrace process */
        char *fname,
        int nproc
)
{
  rtrace_setup(nproc);
  unsigned long  vcount = (hresolu > 1) ? (unsigned long)hresolu*vresolu
                                        : (unsigned long)vresolu;
  long  nextflush = (!vresolu | (hresolu <= 1)) * hresolu;
  int  something2flush = 0;
  FILE  *fp;
  double	d;
  FVECT  orig, direc;
  /* set up input */
  if (fname == NULL)
    inpfp = stdin;
  else if ((inpfp = fopen(fname, "r")) == NULL) {
    sprintf(errmsg, "cannot open input file \"%s\"", fname);
    error(SYSTEM, errmsg);
  }
#ifdef getc_unlocked
  flockfile(inpfp);		/* avoid lock/unlock overhead */
  flockfile(stdout);
#endif
  if (inform != 'a')
          SET_FILE_BINARY(inpfp);
  if (hresolu > 0) {
    if (vresolu > 0)
      fprtresolu(hresolu, vresolu, stdout);
    else
      fflush(stdout);
  }
  /* process input rays */
  while ((d = nextray(orig, direc)) >= 0.0) {
    if (d == 0.0) {				/* flush request? */
      if (something2flush) {
        if (ray_pnprocs > 1 && ray_fifo_flush() < 0)
          error(USER, "child(ren) died");
        bogusray();
        fflush(stdout);
        nextflush = (!vresolu | (hresolu <= 1)) * hresolu;
        something2flush = 0;
      } else
        bogusray();
    } else {				/* compute and print */
      rtcompute(orig, direc, lim_dist ? d : 0.0);
      if (!--nextflush) {		/* flush if time */
        if (ray_pnprocs > 1 && ray_fifo_flush() < 0)
          error(USER, "child(ren) died");
        fflush(stdout);
        nextflush = hresolu;
      } else
        something2flush = 1;
    }
    if (ferror(stdout))
      error(SYSTEM, "write error");
    if (vcount && !--vcount)		/* check for end */
      break;
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
  if (vcount)
    error(WARNING, "unexpected EOF on input");
  if (fflush(stdout) < 0)
    error(SYSTEM, "write error");
  if (fname != NULL) {
    fclose(inpfp);
    inpfp = NULL;
  }
  nextray(NULL, NULL);
#ifdef getc_unlocked
  funlockfile(stdout);
#endif
  ambdone();
}


int
setoutput2(char  *vs, char of)			/* provides additional outputspec Z to output radiance (1 component)*/
{
  oputf_t **table = ray_out;
  int  ncomp = 0;

  if (!*vs)
    error(USER, "empty output specification");

  castonly = 1;			/* sets castonly as side-effect */
  do
    switch (*vs) {
      case 'T':				/* trace sources */
        if (!vs[1]) break;
        trace_sources();
        /* fall through */
      case 't':				/* trace */
        if (!vs[1]) break;
        *table = NULL;
        table = every_out;
        trace = ourtrace;
        castonly = 0;
        break;
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
        if (ambounce > 0 && (ambacc > FTINY || ambssamp > 0))
          error(WARNING,
                "-otV accuracy depends on -aa 0 -as 0");
        break;
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
      case 's':				/* surface */
        *table++ = oputs;
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
        if (ambounce > 0 && (ambacc > FTINY) | (ambssamp > 0))
          error(WARNING,
                "-otW accuracy depends on -aa 0 -as 0");
        break;
      case 'm':				/* modifier */
        *table++ = oputm;
        ncomp++;
        break;
      case 'M':				/* material */
        *table++ = oputM;
        ncomp++;
        break;
      case '~':				/* tilde */
        *table++ = oputtilde;
        break;
      case 'Z':
        *table++ = oputrad;
        castonly = 0;
        ncomp++;
        break;
      default:
        sprintf(errmsg, "unrecognized output option '%c'", *vs);
        error(USER, errmsg);
    }
  while (*++vs);

  *table = NULL;
  if (*every_out != NULL)
    ncomp = 0;
  /* compatibility */
  if ((do_irrad | imm_irrad) && castonly)
    error(USER, "-I+ and -i+ options require some value output");
  for (table = ray_out; *table != NULL; table++) {
    if ((*table == oputV) | (*table == oputW))
      error(WARNING, "-oVW options require trace mode");
    if ((do_irrad | imm_irrad) &&
        (*table == oputr) | (*table == oputR) |
        (*table == oputx) | (*table == oputX))
      error(WARNING, "-orRxX options incompatible with -I+ and -i+");
  }
  switch (of) {
    case 'z': break;
    case 'a': putreal = puta; break;
    case 'f': putreal = putf; break;
    case 'd': putreal = putd; break;
    case 'c':
      if (outvals[1] || !strchr("vrx", outvals[0]))
        error(USER, "color format only with -ov, -or, -ox");
      putreal = putrgbe; break;
    default:
      error(CONSISTENCY, "botched output format");
  }


  return(ncomp);
}

static void
oputrad(				/* print value -o spec: Z*/
        RAY  *r
)
{
  RREAL	lum = bright(r->rcol);
  (*putreal)(&lum, 1);
}

#ifdef __cplusplus
}
#endif
