/* RCSid $Id: func.h,v 2.10 2015/05/20 12:58:31 greg Exp $ */
/*
 * Header file for modifiers using function files.
 *
 * Include after ray.h
 */
#ifndef _RAD_RTINIT_H_
#define _RAD_RTINIT_H_

#ifdef __cplusplus
extern "C" {
#endif


static void onsig(int  signo);
static void sigdie(int  signo, char  *msg);
static void printdefaults(void);
extern int rtinit(int  argc, char  **argv);
extern void rtrace_setup(int nproc);
extern void rtrace_call(char *fname);

#ifdef __cplusplus
}
#endif
#endif /* _RAD_RTMAIN_H_ */

