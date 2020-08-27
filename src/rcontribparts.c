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
 *  rcontribparts.c - partial modified rcontrib.c  (includes at top),
 *  splits call to rcontrib (as rcontrib_call)
 *  to return without exiting and to read input from a file instead of stdin.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "rcontrib.c"
#include "rcinit.h"


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

  if (nproc > 1 && in_rchild())	/* forked child? */{
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
//  quit(0);			/* parent musn't return! */
}



void rcontrib_call(char *fname){
  static int	ignore_warning_given = 0;
  FVECT		orig, direc;
  double		d;
  FILE *fp;
  fp = rcinit2(fname);
  if (fp == NULL){
//    end_children(0);
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
}

void rcontrib_clear(){
  lu_done(&ofiletab);		/* close output files */
//  lu_doall(&modconttab, &lu_delete, NULL);
  for (int i = 0; i < nmods; i++)
    lu_delete(&modconttab, modname[i]);
  nmods = 0;
}



#ifdef __cplusplus
}
#endif
