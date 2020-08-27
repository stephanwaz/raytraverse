/* RCSid $Id: pmapparm.h,v 2.9 2018/02/02 19:47:55 rschregle Exp $ */

/* 
   ======================================================================
   Parameters for photon map generation and rendering; used by mkpmap
   and rpict/rvu/rtrace.   
   
   Roland Schregle (roland.schregle@{hslu.ch, gmail.com}
   (c) Fraunhofer Institute for Solar Energy Systems,
   (c) Lucerne University of Applied Sciences and Arts,
       supported by the Swiss National Science Foundation (SNSF, #147053)
   ======================================================================
   
   $Id: pmapparm.h,v 2.9 2018/02/02 19:47:55 rschregle Exp $
*/


#ifndef PMAPPARAMS_H
   #define PMAPPARAMS_H

   #include "pmaptype.h"
   
      
   /* Struct for passing params per photon map from rpict/rtrace/rvu */
   typedef struct {
      char *fileName;                /* Photon map file */
      unsigned minGather, maxGather; /* Num photons to gather */
      unsigned long distribTarget;   /* Num photons to store */
   } PhotonMapParams;
   
   /* Bounding box for region of interest */
      typedef struct {
      float min [3], max [3];
   } PhotonMapROI;


   extern PhotonMapParams pmapParams [NUM_PMAP_TYPES];
   
   /* Macros for type specific photon map parameters */
   #define globalPmapParams	(pmapParams [PMAP_TYPE_GLOBAL])
   #define preCompPmapParams  (pmapParams [PMAP_TYPE_PRECOMP])
   #define causticPmapParams	(pmapParams [PMAP_TYPE_CAUSTIC])
   #define volumePmapParams	(pmapParams [PMAP_TYPE_VOLUME])
   #define directPmapParams	(pmapParams [PMAP_TYPE_DIRECT])
   #define contribPmapParams  (pmapParams [PMAP_TYPE_CONTRIB])
   
   
   extern float         pdfSamples, preDistrib, finalGather,
                        gatherTolerance, maxDistFix, pmapMaxDist,
                        photonMaxDist;
   extern unsigned long photonHeapSizeInc, photonMaxBounce; 
   extern unsigned      photonRepTime, maxPreDistrib, defaultGather,
                        verbose;

   extern unsigned      pmapNumROI;
   extern PhotonMapROI  *pmapROI;

#ifdef PMAP_OOC
   extern float         pmapCachePageSize;
   extern unsigned long pmapCacheSize;
#endif   


   struct PhotonMap;
   
   int setPmapParam (struct PhotonMap **pm, const PhotonMapParams *parm);
   /* Allocate photon map and set its parameters from parm */
   
   unsigned long parseMultiplier (const char *num);
   /* Evaluate numeric parameter string with optional multiplier suffix
      (G = 10^9, M = 10^6, K = 10^3). Returns 0 if parsing fails. */
#endif
