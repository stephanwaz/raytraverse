#ifdef __cplusplus
extern "C" {
#endif

#include "rtrace.c"
#include "rtinit.h"

extern void
rtrace_setup(				/* initialize processes */
	int  nproc
)
{
	long  nextflush = (!vresolu | (hresolu <= 1)) * hresolu;
	if (imm_irrad)
		castonly = 0;
	else if (castonly)
		nproc = 1;		/* don't bother multiprocessing */
	if ((nextflush > 0) & (nproc > nextflush)) {
		error(WARNING, "reducing number of processes to match flush interval");
		nproc = nextflush;
	}
	if (nproc > 1) {		/* start multiprocessing */
		ray_popen(nproc);
		ray_fifo_out = printvals;
	}
  /* set up output */
  setoutput2(outvals, outform);
}

extern void
rtrace_call(				/* run rtrace process */
        char *fname
)
{
  unsigned long  vcount = (hresolu > 1) ? (unsigned long)hresolu*vresolu
                                        : (unsigned long)vresolu;
  long  nextflush = (!vresolu | (hresolu <= 1)) * hresolu;
  int  something2flush = 0;
  FILE  *fp;
  double	d;
  FVECT  orig, direc;
  /* set up input */
  if (fname == NULL)
    fp = stdin;
  else if ((fp = fopen(fname, "r")) == NULL) {
    sprintf(errmsg, "cannot open input file \"%s\"", fname);
    error(SYSTEM, errmsg);
  }
  if (inform != 'a')
    SET_FILE_BINARY(fp);
  if (hresolu > 0) {
    if (vresolu > 0)
      fprtresolu(hresolu, vresolu, stdout);
    else
      fflush(stdout);
  }
  /* process file */
  while (getvec(orig, inform, fp) == 0 &&
         getvec(direc, inform, fp) == 0) {

    d = normalize(direc);
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
      /* flush if time */
      if (!--nextflush) {
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
  }
  if (fflush(stdout) < 0)
    error(SYSTEM, "write error");
  if (vcount)
    error(USER, "unexpected EOF on input");
  if (fname != NULL)
    fclose(fp);
}


extern void
setoutput2( /* set up output tables */
        char  *vs,
        char of
)
{
  oputf_t **table = ray_out;

  castonly = 1;
  while (*vs)
    switch (*vs++) {
      case 'T':				/* trace sources */
        if (!*vs) break;
        trace_sources();
        /* fall through */
      case 't':				/* trace */
        if (!*vs) break;
        *table = NULL;
        table = every_out;
        trace = ourtrace;
        castonly = 0;
        break;
      case 'o':				/* origin */
        *table++ = oputo;
        break;
      case 'd':				/* direction */
        *table++ = oputd;
        break;
      case 'r':				/* reflected contrib. */
        *table++ = oputr;
        castonly = 0;
        break;
      case 'R':				/* reflected distance */
        *table++ = oputR;
        castonly = 0;
        break;
      case 'x':				/* xmit contrib. */
        *table++ = oputx;
        castonly = 0;
        break;
      case 'X':				/* xmit distance */
        *table++ = oputX;
        castonly = 0;
        break;
      case 'v':				/* value */
        *table++ = oputv;
        castonly = 0;
        break;
      case 'V':				/* contribution */
        *table++ = oputV;
        if (ambounce > 0 && (ambacc > FTINY || ambssamp > 0))
          error(WARNING,
                "-otV accuracy depends on -aa 0 -as 0");
        break;
      case 'l':				/* effective distance */
        *table++ = oputl;
        castonly = 0;
        break;
      case 'c':				/* local coordinates */
        *table++ = oputc;
        break;
      case 'L':				/* single ray length */
        *table++ = oputL;
        break;
      case 'p':				/* point */
        *table++ = oputp;
        break;
      case 'n':				/* perturbed normal */
        *table++ = oputn;
        castonly = 0;
        break;
      case 'N':				/* unperturbed normal */
        *table++ = oputN;
        break;
      case 's':				/* surface */
        *table++ = oputs;
        break;
      case 'w':				/* weight */
        *table++ = oputw;
        break;
      case 'W':				/* coefficient */
        *table++ = oputW;
        castonly = 0;
        if (ambounce > 0 && (ambacc > FTINY) | (ambssamp > 0))
          error(WARNING,
                "-otW accuracy depends on -aa 0 -as 0");
        break;
      case 'm':				/* modifier */
        *table++ = oputm;
        break;
      case 'M':				/* material */
        *table++ = oputM;
        break;
      case '~':				/* tilde */
        *table++ = oputtilde;
        break;
      case 'Z': /* radiance */
        *table++ = oputrad;
        castonly = 0;
        break;
    }
  *table = NULL;
  /* compatibility */
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
      if (outvals[0] && (outvals[1] || !strchr("vrx", outvals[0])))
        error(USER, "color format only with -ov, -or, -ox");
      putreal = putrgbe; break;
    default:
      error(CONSISTENCY, "botched output format");
  }
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
