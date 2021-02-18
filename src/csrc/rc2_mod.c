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
#ifndef lint
static const char RCSid[] = "$Id: rc2.c,v 2.24 2019/09/04 20:19:51 greg Exp $";
#endif
/*
 * Accumulate ray contributions for a set of materials
 * File i/o and recovery
 */

#include <ctype.h>
#include "platform.h"
#include "rcontrib.h"
#include "resolu.h"
#include "rcinit.h"


RREAL *output_values;

/* Close output stream and free record */
static void
closestream(void *p)
{
  STREAMOUT	*sop = (STREAMOUT *)p;

  if (sop->ofp != NULL) {
    int	status = 0;
    if (sop->outpipe)
      status = pclose(sop->ofp);
    else if (sop->ofp != stdout)
      status = fclose(sop->ofp);
    if (status)
      error(SYSTEM, "error closing output stream");
  }
  free(p);
}

LUTAB	ofiletab = LU_SINIT(free,closestream);	/* output file table */

#define OF_MODIFIER	01
#define OF_BIN		02

/************************** STREAM & FILE I/O ***************************/


/* Get output stream pointer (only stdout, else returns null*/
FILE *
getostream2(const char *ospec, int noopen) {

  if (ospec == NULL) {      /* use stdout? */
    if (!noopen & !using_stdout) {
       SET_FILE_BINARY(stdout);
#ifdef getc_unlocked
      flockfile(stdout);  /* avoid lock/unlock overhead */
#endif
      if (waitflush > 0)
        fflush(stdout);
      using_stdout = 1;
    }
    return stdout;
  }
  return NULL;
}

void putn(RREAL *v, int n){ /* output to buffer */
  for (int i = 0; i < n; i++){
    output_values[putcount + i] = v[i];
  }
  putcount += n;
}

/* Put out ray contribution to file */
void
put_contrib(const DCOLOR cnt, FILE *fout)
{
  double	sf = 1;
  if (accumulate > 1)
    sf = 1./(double)accumulate;
  if (fout != NULL){
    DCOLOR	dv;
    copycolor(dv, cnt);
    scalecolor(dv, sf);
    putbinary(dv, sizeof(double), 3, fout);
  } else if (outbright) {
      double lum = bright(cnt) * sf;
      putn(&lum, 1);
  } else {
      DCOLOR	dv;
      copycolor(dv, cnt);
      scalecolor(dv, sf);
      putn(dv, 3);
  }
}

/* Output modifier values to appropriate stream(s) */
void
mod_output(MODCONT *mp)
{
  FILE	*sop = getostream2(mp->outspec, 0);
  int		j;
  for (int j = 0; j < mp->nbins; j++) {
    put_contrib(mp->cbin[j], sop);
  }
}

/* callback to flush as requested */
static int
puteol(const LUENT *e, void *p)
{
  STREAMOUT	*sop = (STREAMOUT *)e->data;

  if (!waitflush)
    fflush(sop->ofp);
  if (ferror(sop->ofp)) {
    sprintf(errmsg, "write error on file '%s'", e->key);
    error(SYSTEM, errmsg);
  }
  return(0);
}

/* Terminate record output and flush if time */
void
end_record()
{
  --waitflush;
  lu_doall(&ofiletab, &puteol, NULL);
  if (!waitflush) {
    waitflush = (yres > 0) & (xres > 1) ? 0 : xres;
    if (using_stdout)
      fflush(stdout);
  }
}

