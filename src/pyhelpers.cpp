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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

const char* docstring = R"pbdoc(raytraverse helper functions written in c++)pbdoc";

const char* from_pdf_docstring =R"pbdoc(helper function for draw.from_pdf

Parameters
----------
pdf: np.array
    array of doubles with weights to check against threshold
candidates: np.array
    array of integers to write candidate indices to, should be atleast as large as pdf
bidx: np.array
    array of integers to write indices above upper bound, should be atleast as large as pdf
threshold: float
    value used to determine the number of indices to return
lb: float, optional
    values below threshold * lb will be excluded from candidates (lb must be in (0,1)
ub: float, optional
    values above threshold * ub will have indices written to bidx

Returns
-------
ccnt: int
    the number of choices (use to slice end of candidates
bcnt: int
    the number of indices above the upper bound, use to slice bidx
nsampc: int
    the number of draws that should be selected from the candidates
    )pbdoc";

pybind11::tuple from_pdf(py::array_t<double> &pdf,
                         py::array_t<u_int32_t> candidates,
                         py::array_t<u_int32_t> bidx,
                         double threshold, double lb, double ub) {
  auto parr = pdf.unchecked<1>();
  auto carr = candidates.mutable_unchecked<1>();
  auto barr = bidx.mutable_unchecked<1>();
  u_int32_t ccnt = 0, bcnt = 0, nsampc = 0, cumsum = 0;

  for (size_t idx = 0; idx < parr.shape(0); idx++) {
    // cliphigh
    if (parr(idx) < threshold * ub){
      // clip: set number of samples
      if (parr(idx) > threshold){
        nsampc++;
      }
      // cliplow: add to candidates and update probability
      if (parr(idx) > threshold * lb){
        carr(ccnt) = cumsum + ccnt;
        ccnt++;
      }
        // add to index offset
      else {
        cumsum++;
      }
      // definitely draw and add to index offset
    } else {
      barr(bcnt) = idx;
      bcnt++;
      cumsum++;
    }
  }
  return py::make_tuple(ccnt, bcnt, nsampc);
}

//const char* get_posidx_docstring =R"pbdoc(get the position index of a ray relative to a view direction
//
//Parameters
//----------
//vx: double
//  view x
//vy: double
//  view y
//vz: double
//  view z
//dx: double
//  ray x
//dy: double
//  ray y
//dz: double
//  ray z
//postype: int
//  position index type (1=KIM, else guth/iwata)
//
//Returns
//-------
//posidx: float)pbdoc";
//
//double get_posidx(double vx, double vy, double vz, double dx, double dy, double dz, int postype) {
//  float deg = 180 / 3.1415927;
//  float fact = 0.8;
//
//  double vm = std::sqrt(vx*vx + vy*vy + vz*vz);
//  double dm = std::sqrt(dx*dx + dy*dy + dz*dz);
//  double sigma = std::acos((vx*dx + vy*dy + vz*dz)/(vm*dm));
//  double phi =
//
//  if (phi == 0) {
//    phi = 0.00001;
//  }
//  if (sigma <= 0) {
//    sigma = -sigma;
//
//  if (theta == 0) {
//    theta = 0.0001;
//  }
//  tau = tau * deg;
//  sigma = sigma * deg;
//
//  if (postype == 1) {
///* KIM  model */
//    posindex = exp ((sigma-(-0.000009*tau*tau*tau+0.0014*tau*tau+0.0866*tau+21.633))/(-0.000009*tau*tau*tau+0.0013*tau*tau+0.0853*tau+8.772));
//  }else{
///* Guth model, equation from IES lighting handbook */
//    posindex =
//            exp((35.2 - 0.31889 * tau -
//                 1.22 * exp(-2 * tau / 9)) / 1000 * sigma + (21 +
//                                                             0.26667 * tau -
//                                                             0.002963 * tau *
//                                                             tau) / 100000 *
//                                                            sigma * sigma);
///* below line of sight, using Iwata model */
//    if (phi < 0) {
//      d = 1 / tan(phi);
//      s = tan(teta) / tan(phi);
//      r = sqrt(1 / d * 1 / d + s * s / d / d);
//      if (r > 0.6)
//        fact = 1.2;
//      if (r > 3) {
//        fact = 1.2;
//        r = 3;
//      }
//
//      posindex = 1 + fact * r;
//    }
//    if (posindex > 16)
//      posindex = 16;
//  }
//
//  return posindex;
//}

using namespace pybind11::literals;
PYBIND11_MODULE(craytraverse, m) {
  m.doc() = docstring;
  m.def("from_pdf", &from_pdf,
        "pdf"_a,
        "candidates"_a,
        "bidx"_a,
        "threshold"_a,
        "lb"_a=.5,
        "ub"_a=4.0,
        from_pdf_docstring);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
