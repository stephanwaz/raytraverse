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
  m.def("f", []() {
      // Allocate and initialize some data; make this big so
      // we can see the impact on the process memory use:
      constexpr size_t size = 100*1000*1000;
      double *foo = new double[size];
      for (size_t i = 0; i < size; i++) {
        foo[i] = (double) i;
      }

      // Create a Python object that will free the allocated
      // memory when destroyed:
      py::capsule free_when_done(foo, [](void *f) {
          double *foo = reinterpret_cast<double *>(f);
          std::cerr << "Element [0] = " << foo[0] << "\n";
          std::cerr << "freeing memory @ " << f << "\n";
          delete[] foo;
      });

      return py::array_t<double>(
              {100, 1000, 1000}, // shape
              {1000*1000*8, 1000*8, 8}, // C-style contiguous strides for double
              foo, // the data pointer
              free_when_done); // numpy array references this parent
  });

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
