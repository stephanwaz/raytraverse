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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

const char* docstring = R"pbdoc(raytraverse helper functions written in c++)pbdoc";

const char* from_pdf_docstring =R"pbdoc(helper function for draw.from_pdf

Parameters
----------
pdf: np.array
    array of doubles with weights to check against threshold

threshold: float
    value used to determine the number of indices to return
lb: float, optional
    values below threshold * lb will be excluded from candidates (lb must be in (0,1)
ub: float, optional
    values above threshold * ub will have indices written to bidx

Returns
-------
candidates: np.array
    array of candidate indices
bidx: np.array
    array of definitely included indices
nsampc: int
    the number of draws that should be selected from the candidates
    )pbdoc";

pybind11::tuple from_pdf(py::array_t<double> &pdf, double threshold, double lb, double ub) {

  // initialize outputs at largest possible size
  py::array_t<u_int32_t> candidates(pdf.size());
  py::buffer_info cout = candidates.request();
  auto *carr = (u_int32_t *) cout.ptr;

  py::array_t<u_int32_t> bidx(pdf.size());
  py::buffer_info bout = bidx.request();
  auto *barr = (u_int32_t *) bout.ptr;

  u_int32_t ccnt = 0, bcnt = 0, nsampc = 0, cumsum = 0;

  for (size_t idx = 0; idx < pdf.shape(0); idx++) {
    // cliphigh
    if (pdf.at(idx) < threshold * ub){
      // clip: set number of samples
      if (pdf.at(idx) > threshold){
        nsampc++;
      }
      // cliplow: add to candidates and update probability
      if (pdf.at(idx) > threshold * lb){
        carr[ccnt] = cumsum + ccnt;
        ccnt++;
      }
        // add to index offset
      else {
        cumsum++;
      }
      // definitely draw and add to index offset
    } else {
      barr[bcnt] = idx;
      bcnt++;
      cumsum++;
    }
  }
  // trim to used size before returning (frees memory when relinquished to python)
  candidates.resize({ccnt});
  bidx.resize({bcnt});
  return py::make_tuple(candidates, bidx, nsampc);
}


using namespace pybind11::literals;
PYBIND11_MODULE(craytraverse, m) {
  m.doc() = docstring;
  m.def("from_pdf", &from_pdf,
        "pdf"_a,
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
