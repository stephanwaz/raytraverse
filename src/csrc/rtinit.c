#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-macro-parentheses"
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
 *  rtinit.c - modified main routine from rtmain.c called rtinit,
 *  to load command in to memory but not execute or exit.
 */


#ifdef __cplusplus
extern "C" {
#endif

#include "copyright.h"

#include  <signal.h>

#include  "platform.h"
#include  "ray.h"
#include  "source.h"
#include  "ambient.h"
#include  "random.h"
#include  "paths.h"
#include  "pmapray.h"
#include "rtinit.h"

extern char	*progname;		/* global argv[0] */

/* persistent processes define */
#ifdef  F_SETLKW
#define  PERSIST	1		/* normal persist */
#define  PARALLEL	2		/* parallel persist */
#define  PCHILD		3		/* child of normal persist */
#endif

char  *sigerr[NSIG];			/* signal error messages */
char  *errfile = NULL;			/* error output file */

int  nproc = 1;				/* number of processes */


char  *outvals = "Z";			/* output specification */

int  hresolu = 0;			/* horizontal (scan) size */
int  vresolu = 0;			/* vertical resolution */

extern int  castonly;			/* only doing ray-casting? */

extern int  imm_irrad;			/* compute immediate irradiance? */
int  lim_dist = 0;			/* limit distance? */

#ifndef	MAXMODLIST
#define	MAXMODLIST	1024		/* maximum modifiers we'll track */
#endif


static int  loadflags = ~IO_FILES;	/* what to load from octree */

int repeat = 1;

void
eputsrt(				/* put string to stderr */
        register char  *s
)
{
  static int  midline = 0;

  if (!*s)
    return;
  if (!midline++) {
    fputs(progname, stderr);
    fputs(": ", stderr);
  }
  fputs(s, stderr);
  if (s[strlen(s)-1] == '\n') {
    fflush(stderr);
    midline = 0;
  }
}

void
wputsrt(				/* warning output function */
        char	*s
)
{
  int  lasterrno = errno;
  eputsrt(s);
  errno = lasterrno;
}


int
rtinit(int  argc, char  *argv[])
{
#define	 check(ol,al)		if (argv[i][ol] || \
				badarg(argc-i-1,argv+i+1,al)) \
				goto badopt
#define	 check_bool(olen,var)		switch (argv[i][olen]) { \
				case '\0': (var) = !var; break; \
				case 'y': case 'Y': case 't': case 'T': \
				case '+': case '1': var = 1; break; \
				case 'n': case 'N': case 'f': case 'F': \
				case '-': case '0': var = 0; break; \
				default: goto badopt; }
  int complete = 0;
  int  rval;
  int  i;
  repeat = 1;
  ambdone();
  freeqstr(ambfile);
  ambfile = NULL;
  /* global program name */
  progname = argv[0] = fixargv0(argv[0]);
  /* add trace notify function */
  for (i = 0; addobjnotify[i] != NULL; i++) {
    addobjnotify[i] = NULL;
  }
  addobjnotify[0] = ambnotify;
  /* option city */
  /* reset these with each call */
  imm_irrad = 0;			/* compute immediate irradiance? */
  lim_dist = 0;			/* limit distance? */
  outvals = "Z";			/* output specification */
  loadflags = ~IO_FILES;
  for (i = 1; i < argc; i++) {
    /* expand arguments */
    while ((rval = expandarg(&argc, &argv, i)) > 0)
      ;
    if (rval < 0) {
      sprintf(errmsg, "cannot expand '%s'", argv[i]);
      goto badopt;
    }
    if (argv[i] == NULL || argv[i][0] != '-')
      break;			/* break from options */
    rval = getrenderopt(argc-i, argv+i);
    if (rval >= 0) {
      i += rval;
      continue;
    }
    switch (argv[i][1]) {
      case 'c':
        check(2,"i");
        repeat = atoi(argv[++i]);
        break;
      case 'n':				/* number of cores */
        check(2,"i");
        nproc = atoi(argv[++i]);
        if (nproc <= 0)
          error(USER, "bad number of processes");
        break;
      case 'w':				/* warnings */
        rval = erract[WARNING].pf != NULL;
        check_bool(2,rval);
        if (rval) erract[WARNING].pf = wputsrt;
        else erract[WARNING].pf = NULL;
        break;
      case 'e':				/* error file */
        check(2,"s");
        errfile = argv[++i];
        break;
      case 'l':				/* limit distance */
        if (argv[i][2] != 'd')
          goto badopt;
        check_bool(3,lim_dist);
        break;
      case 'I':				/* immed. irradiance */
        check_bool(2,imm_irrad);
        break;
      case 'o':				/* output */
        outvals = argv[i]+2;
        break;
      default:
        goto badopt;
    }
  }
  /* initialize object types */
  initotypes();
  /* initialize urand */
  if (rand_samp) {
    srandom((long)time(0));
    initurand(0);
  } else {
    srandom(0L);
    initurand(2048);
  }
  /* set up signal handling */
  sigdie(SIGINT, "Interrupt");
#ifdef SIGHUP
  sigdie(SIGHUP, "Hangup");
#endif
  sigdie(SIGTERM, "Terminate");
#ifdef SIGPIPE
  sigdie(SIGPIPE, "Broken pipe");
#endif
#ifdef SIGALRM
  sigdie(SIGALRM, "Alarm clock");
#endif
#ifdef	SIGXCPU
  sigdie(SIGXCPU, "CPU limit exceeded");
  sigdie(SIGXFSZ, "File size exceeded");
#endif
  /* open error file */
  if (errfile != NULL) {
    if (freopen(errfile, "a", stderr) == NULL)
      quit(2);
    fprintf(stderr, "**************\n*** PID %5d: ",
            getpid());
    printargs(argc, argv, stderr);
    putc('\n', stderr);
    fflush(stderr);
  }
#ifdef	NICE
  nice(NICE);			/* lower priority */
#endif
  /* get octree */
  if (i != argc)
    goto badopt;
  complete = setoutput2(outvals);
  badopt:
  if (complete < 1){
    return -1;
  }

  return nproc;

#undef	check
#undef	check_bool
}

void
rtrace_loadscene(char* pyoctnm) {
  /* get octree */
  char octnm[strlen(pyoctnm)];
  strcpy(octnm, pyoctnm);
  extern char  *octname;
  readoct(octname = octnm, loadflags & ~IO_INFO, &thescene, NULL);
  octname = NULL;
  nsceneobjs = nobjects;
}

int
rtrace_loadsrc(char* srcname, int freesrc) {
  int oldcnt = nobjects;
  ambnotify(OVOID);
  freesources();
  if (freesrc > 0) {
    freeobjects(nobjects - freesrc, freesrc);
  }
  if (srcname != NULL) {
    readobj(srcname);
    nsceneobjs = nobjects;
  }
  /* don't warn about sources as this is expected behavior */
  int warnings = erract[WARNING].pf != NULL;
  erract[WARNING].pf = NULL;
  if (!castonly) {	/* any actual ray traversal to do? */
    ray_init_pmap();	/* PMAP: set up & load photon maps */
    marksources();		/* find and mark sources */
  } else
    distantsources();	/* else mark only distant sources */
  if (warnings) {
    erract[WARNING].pf = wputsrt;
  }
  return nobjects - oldcnt;
}


static void
onsig(				/* fatal signal */
        int  signo
)
{
  static int  gotsig = 0;

  if (gotsig++)			/* two signals and we're gone! */
    _exit(signo);

#ifdef SIGALRM
  alarm(15);			/* allow 15 seconds to clean up */
  signal(SIGALRM, SIG_DFL);	/* make certain we do die */
#endif
  eputsrt("signal - ");
  eputsrt(sigerr[signo]);
  eputsrt("\n");
  quit(3);
}


static void
sigdie(			/* set fatal signal */
        int  signo,
        char  *msg
)
{
  if (signal(signo, onsig) == SIG_IGN)
    signal(signo, SIG_IGN);
  sigerr[signo] = msg;
}

#ifdef __cplusplus
}
#endif

#pragma clang diagnostic pop