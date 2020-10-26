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

#include <signal.h>
#include "selcall.h"
#include <rc3.c>
#include "rcinit.h"


// rc3.c - modifies parallel execution from ray/src/rtrc3.c to read input from file.


/* Run parental oversight loop */
void
parental_loop2(char *fname)
{
  const int	qlimit = (accumulate == 1) ? 1 : MAXIQ-1;
  int		ninq = 0;
  FVECT		orgdir[2*MAXIQ];
  int		i, n;
  /* load rays from stdin & process */
  FILE *fp;
  if (fname == NULL)
    fp = stdin;
  else if ((fp = fopen(fname, "rb")) == NULL) {
    fprintf(stderr, "help!");
    sprintf(errmsg, "cannot open input file \"%s\"", fname);
    error(SYSTEM, errmsg);
  }
#ifdef getc_unlocked
  flockfile(fp);		/* avoid lock/unlock overhead */
#endif
  while (getvecfp(orgdir[2*ninq], fp) == 0 && getvecfp(orgdir[2*ninq+1], fp) == 0) {
    const int	zero_ray = orgdir[2*ninq+1][0] == 0.0 &&
                          (orgdir[2*ninq+1][1] == 0.0) &
                          (orgdir[2*ninq+1][2] == 0.0);
    ninq += !zero_ray;
    /* Zero ray cannot go in input queue */
    if (zero_ray ? ninq : ninq >= qlimit ||
                          lastray/accumulate != (lastray+ninq)/accumulate) {
      i = next_child_nq(0);		/* manages output */
      n = ninq;
      if (accumulate > 1)		/* need terminator? */
        memset(orgdir[2*n++], 0, sizeof(FVECT)*2);
      n *= sizeof(FVECT)*2;		/* send assignment */
      if (writebuf(kidpr[i].w, (char *)orgdir, n) != n)
        error(SYSTEM, "pipe write error");
      kida[i].r1 = lastray+1;
      lastray += kida[i].nr = ninq;	/* mark as busy */
      if (lastray < lastdone) {	/* RNUMBER wrapped? */
        while (next_child_nq(1) >= 0)
          ;
        lastray -= ninq;
        lastdone = lastray %= accumulate;
      }
      ninq = 0;
    }
    if (zero_ray) {				/* put bogus record? */
      if ((yres <= 0) | (xres <= 1) &&
          (lastray+1) % accumulate == 0) {
        while (next_child_nq(1) >= 0)
          ;		/* clear the queue */
        lastdone = lastray = accumulate-1;
        waitflush = 1;		/* flush next */
      }
      put_zero_record(++lastray);
    }
    if (raysleft && !--raysleft)
      break;				/* preemptive EOI */
  }
  while (next_child_nq(1) >= 0)		/* empty results queue */
    ;
  if (account < accumulate) {
    error(WARNING, "partial accumulation in final record");
    free_binq(out_bq);		/* XXX just ignore it */
    out_bq = NULL;
  }
#ifdef getc_unlocked
  funlockfile(fp);		/* avoid lock/unlock overhead */
#endif
  fclose(fp);
  free_binq(NULL);			/* clean up */
  lu_done(&ofiletab);
  end_children(0); /* wait for children */
  nchild = 0; /* reset child count in case of future call*/
  if (raysleft)
    error(USER, "unexpected EOF on input");
}

void
feeder_loop2(char *fname)
{
  static int	ignore_warning_given = 0;
  int		ninq = 0;
  FVECT		orgdir[2*MAXIQ];
  int		i, n;
  /* load rays from stdin & process */
  FILE *fp;
  if (fname == NULL)
    fp = stdin;
  else if ((fp = fopen(fname, "rb")) == NULL) {
    sprintf(errmsg, "cannot open input file \"%s\"", fname);
    error(SYSTEM, errmsg);
  }
#ifdef getc_unlocked
  flockfile(fp);		/* avoid lock/unlock overhead */
#endif
  while (getvecfp(orgdir[2*ninq], fp) == 0 && getvecfp(orgdir[2*ninq+1], fp) == 0) {
    if (orgdir[2*ninq+1][0] == 0.0 &&	/* asking for flush? */
        (orgdir[2*ninq+1][1] == 0.0) &
        (orgdir[2*ninq+1][2] == 0.0)) {
      if (!ignore_warning_given++)
        error(WARNING,
              "dummy ray(s) ignored during accumulation\n");
      continue;
    }
    if (++ninq >= MAXIQ) {
      i = next_child_ready();		/* get eager child */
      n = sizeof(FVECT)*2 * ninq;	/* give assignment */
      if (writebuf(kidpr[i].w, (char *)orgdir, n) != n)
        error(SYSTEM, "pipe write error");
      kida[i].r1 = lastray+1;
      lastray += kida[i].nr = ninq;
      if (lastray < lastdone)		/* RNUMBER wrapped? */
        lastdone = lastray = 0;
      ninq = 0;
    }
    if (raysleft && !--raysleft)
      break;				/* preemptive EOI */
  }
  if (ninq) {				/* polish off input */
    i = next_child_ready();
    n = sizeof(FVECT)*2 * ninq;
    if (writebuf(kidpr[i].w, (char *)orgdir, n) != n)
      error(SYSTEM, "pipe write error");
    kida[i].r1 = lastray+1;
    lastray += kida[i].nr = ninq;
    ninq = 0;
  }
  memset(orgdir, 0, sizeof(FVECT)*2);	/* get results */
  for (i = nchild; i--; ) {
    writebuf(kidpr[i].w, (char *)orgdir, sizeof(FVECT)*2);
    queue_results(i);
  }
  if (recover)				/* and from before? */
    queue_modifiers();
  end_children(0);			/* free up file descriptors */
  for (i = 0; i < nmods; i++)
    mod_output(out_bq->mca[i]);	/* output accumulated record */
  end_record();
  free_binq(out_bq);			/* clean up */
  out_bq = NULL;
  free_binq(NULL);
  lu_done(&ofiletab);
#ifdef getc_unlocked
  funlockfile(fp);		/* avoid lock/unlock overhead */
#endif
  fclose(fp);
  if (raysleft)
    error(USER, "unexpected EOF on input");
}

/* Start child processes if we can (call only once in parent!) */
int
in_rchild2()
{
  int	rval;

  while (nchild < nproc) {	/* fork until target reached */
    errno = 0;
    rval = open_process(&kidpr[nchild], NULL);
    if (rval < 0)
      error(SYSTEM, "open_process() call failed");
    if (rval == 0) {	/* if in child, set up & return true */
      lu_doall(&modconttab, &set_stdout, NULL);
      lu_done(&ofiletab);
      while (nchild--) {	/* don't share other pipes */
        close(kidpr[nchild].w);
        fclose(kida[nchild].infp);
      }
      inpfmt = (sizeof(RREAL)==sizeof(double)) ? 'd' : 'f';
      outfmt = 'z'; /* to bybass possible brightness output in parent */
      header = 0;
      yres = 0;
      raysleft = 0;
      if (accumulate == 1) {
        waitflush = xres = 1;
        account = accumulate = 1;
      } else {	/* parent controls accumulation */
        waitflush = xres = 0;
        account = accumulate = 0;
      }
      return(1);	/* return "true" in child */
    }
    if (rval != PIPE_BUF)
      error(CONSISTENCY, "bad value from open_process()");
    /* connect to child's output */
    kida[nchild].infp = fdopen(kidpr[nchild].r, "rb");
    if (kida[nchild].infp == NULL)
      error(SYSTEM, "out of memory in in_rchild()");
    kida[nchild++].nr = 0;	/* mark as available */
  }
#ifdef getc_unlocked
  for (rval = nchild; rval--; )	/* avoid mutex overhead */
    flockfile(kida[rval].infp);
#endif
  return(0);			/* return "false" in parent */
}
