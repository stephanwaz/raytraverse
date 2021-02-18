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

/* Construct output file name and return flags whether modifier/bin present */
static int
ofname(char *oname, const char *ospec, const char *mname, int bn)
{
  const char	*mnp = NULL;
  const char	*bnp = NULL;
  const char	*cp;

  if (ospec == NULL)
    return(-1);
  for (cp = ospec; *cp; cp++)		/* check format position(s) */
    if (*cp == '%') {
      do
        ++cp;
      while (isdigit(*cp));
      switch (*cp) {
        case '%':
          break;
        case 's':
          if (mnp != NULL)
            return(-1);
          mnp = cp;
          break;
        case 'd':
        case 'i':
        case 'o':
        case 'x':
        case 'X':
          if (bnp != NULL)
            return(-1);
          bnp = cp;
          break;
        default:
          return(-1);
      }
    }
  if (mnp != NULL) {			/* create file name */
    if (bnp != NULL) {
      if (bnp > mnp)
        sprintf(oname, ospec, mname, bn);
      else
        sprintf(oname, ospec, bn, mname);
      return(OF_MODIFIER|OF_BIN);
    }
    sprintf(oname, ospec, mname);
    return(OF_MODIFIER);
  }
  if (bnp != NULL) {
    sprintf(oname, ospec, bn);
    return(OF_BIN);
  }
  strcpy(oname, ospec);
  return(0);
}


/* Write header to the given output stream */
static void
printheader(FILE *fout, const char *info)
{
  extern char	VersionID[];
  /* copy octree header */
  if (octname[0] == '!') {
    newheader("RADIANCE", fout);
    fputs(octname+1, fout);
    if (octname[strlen(octname)-1] != '\n')
      fputc('\n', fout);
  } else {
    FILE	*fin = fopen(octname, (outfmt=='a') ? "r" : "rb");
    if (fin == NULL)
      quit(1);
    checkheader(fin, OCTFMT, fout);
    fclose(fin);
  }
  printargs(gargc, gargv, fout);	/* add our command */
  fprintf(fout, "SOFTWARE= %s\n", VersionID);
  fputnow(fout);
  if (outbright)
    fputs("NCOMP=1\n", fout);
  else
    fputs("NCOMP=3\n", fout);		/* always RGB */
  if (info != NULL)			/* add extra info if given */
    fputs(info, fout);
  if ((outfmt == 'f') | (outfmt == 'd'))
    fputendian(fout);
  fputformat(formstr(outfmt), fout);
  fputc('\n', fout);			/* empty line ends header */
}


/* Write resolution string to given output stream */
static void
printresolu(FILE *fout, int xr, int yr)
{
  if ((xr > 0) & (yr > 0))	/* resolution string */
    fprtresolu(xr, yr, fout);
}


/* Get output stream pointer (open and write header if new and noopen==0) */
STREAMOUT *
getostream(const char *ospec, const char *mname, int bn, int noopen) {
  static STREAMOUT stdos;
  char info[1024];
  int ofl;
  char oname[1024];
  LUENT *lep;
  STREAMOUT *sop;
  char *cp;

  if (ospec == NULL) {      /* use stdout? */
    if (!noopen & !using_stdout) {
      if (outfmt != 'a')
              SET_FILE_BINARY(stdout);
#ifdef getc_unlocked
      flockfile(stdout);  /* avoid lock/unlock overhead */
#endif
      if (waitflush > 0)
        fflush(stdout);
      stdos.xr = xres;
      stdos.yr = yres;
      using_stdout = 1;
    }
    stdos.ofp = stdout;
    stdos.reclen += noopen;
    return (&stdos);
  }
  return NULL;
}


/* Get a vector from stdin */
int
getvec(FVECT vec)
{
  float	vf[3];
  double	vd[3];
  char	buf[32];
  int	i;

  switch (inpfmt) {
    case 'a':					/* ascii */
      for (i = 0; i < 3; i++) {
        if (fgetword(buf, sizeof(buf), stdin) == NULL ||
            !isflt(buf))
          return(-1);
        vec[i] = atof(buf);
      }
      break;
    case 'f':					/* binary float */
      if (getbinary((char *)vf, sizeof(float), 3, stdin) != 3)
        return(-1);
      VCOPY(vec, vf);
      break;
    case 'd':					/* binary double */
      if (getbinary((char *)vd, sizeof(double), 3, stdin) != 3)
        return(-1);
      VCOPY(vec, vd);
      break;
    default:
      error(CONSISTENCY, "botched input format");
  }
  return(0);
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


///* Put out ray contribution to file */



/* Output modifier values to appropriate stream(s) */
/* Output modifier values to appropriate stream(s) */
void
mod_output(MODCONT *mp)
{
  STREAMOUT	*sop = getostream(mp->outspec, mp->modname, mp->bin0, 0);
  int		j;
  if (sop == NULL) {
    for (int j = 0; j < mp->nbins; j++) {
      put_contrib(mp->cbin[j], NULL);
    }
  } else {
    put_contrib(mp->cbin[0], sop->ofp);
    if (mp->nbins > 3 &&	/* minor optimization */
        sop == getostream(mp->outspec, mp->modname, mp->bin0+1, 0)) {
      for (j = 1; j < mp->nbins; j++)
        put_contrib(mp->cbin[j], sop->ofp);
    } else {
      for (j = 1; j < mp->nbins; j++) {
        sop = getostream(mp->outspec, mp->modname, mp->bin0+j, 0);
        put_contrib(mp->cbin[j], sop->ofp);
      }
    }
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

