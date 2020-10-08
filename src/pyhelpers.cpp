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



const char* interpolate_kdquery_docstring =R"pbdoc(interpolate luminance values associated with query results
from scipy.cKDTree.query. Finds closest point and then locates vertices of enclosing triangle from this point.
returns 0 in cases where the query provides no results, so the distance_upper_bound must be set appropriately.

Parameters
----------
dest_vec: np.array
    destination vectors to interpolate to, shape (N, 3)
errs: np.array
    distances between src and destination (row matches dest_vec, column is sorted ascending), shape (N, # of queries)
idxs: np.array
    query result, index row in src_vec close to dest_vec, shape (N, # of queries)
src_vec: np.array
    vectors of src_kd, shape (N, 3)
src_lum: np.array
    luminance values for src_kd, shape (src_vec.shape[0], srcn)
err: float, optional
    distance below which closest sample is used directly

Returns
-------
arrout: np.array
    destination luminances shape (N, srcn))pbdoc";

py::array_t<double> interpolate_kdquery(py::array_t<double> &destvec,
                                        py::array_t<double> &errs,
                                        py::array_t<int> &idxs,
                                        py::array_t<double> &srcvec,
                                        py::array_t<double> &srclum,
                                        double err) {
  // get shape information
  auto drows = destvec.shape(0);
  auto dcols = srclum.shape(1);
  auto srows = srclum.shape(0);
  auto qcols = idxs.shape(1);

  //intitialize result
  py::array_t<double> arrout({drows, dcols});
  py::buffer_info out = arrout.request();
  auto *pout = (double *) out.ptr;

  ssize_t qcnt; //valid queries (index in range)
  ssize_t row; //track row position in memory for result
  double nx, ny, nz, nm; //normalization
  int indo[3]; //interpolation indices
  double dt[3]; //interpolation distances
  double nd; // normalize interpolation distances (to barycentric)
  int c, v2; //interpolation count
  double dot; //first dot product
  double dot2; //second dot product
  double cutoff = 0.0; //cos 90 degrees (to block out hemisphere)
  double cutoff2 = sqrt(2)/2; //cos 45 degrees (to block out quadrant)
  double outv; //interpolation result
  // i loops over destination vectors
  // j loops over sources
  // k loops over query results
  for (ssize_t i = 0; i < drows; i++) {
    row = i*dcols;
    // check for number of queries in bounds
    for (qcnt = 0; qcnt < qcols; qcnt++) {
      if (idxs.at(i, qcnt) >= srows)
        break;
    }
    // if none, skip
    if (qcnt == 0){
      for (ssize_t j = 0; j < dcols; j++){
        pout[row + j] = 0;
      }
      continue;
    }
    // if the closes point is within tolerance, just use that
    if (errs.at(i, 0) < err) {
      for (ssize_t j = 0; j < dcols; j++){
        pout[row + j] = srclum.at(idxs.at(i, 0), j);
      }
      continue;
    }

    std::vector<double> ivecs(qcnt * 3);
    //normalize interpolation vectors
    for (ssize_t k = 0; k < qcnt; k++) {
      nx = srcvec.at(idxs.at(i, k), 0) - destvec.at(i, 0);
      ny = srcvec.at(idxs.at(i, k), 1) - destvec.at(i, 1);
      nz = srcvec.at(idxs.at(i, k), 2) - destvec.at(i, 2);
      nm = sqrt(nx * nx + ny * ny + nz * nz);
      ivecs[k*3] = nx / nm;
      ivecs[k*3+1] = ny / nm;
      ivecs[k*3+2] = nz / nm;
    }
    c = 0;
    indo[0] = idxs.at(i, 0);
    dt[0] = 1.0 / pow(errs.at(i, 0), 2);
    //look for second vector 180 degrees from first, and then third 90 degrees from second
    for (ssize_t k = 1; k < qcnt; k++) {
      dot = ivecs.at(0)*ivecs.at(k*3) + ivecs.at(1)*ivecs.at(k*3+1) + ivecs.at(2)*ivecs.at(k*3+2);
      if (dot > cutoff) {
        continue;
      }
      //the first time we make it here, store vertex 2
      if (c == 0){
        c++;
        indo[1] = idxs.at(i, k);
        dt[1] = 1.0 / pow(errs.at(i, k), 2);
        v2 = k;
      }
      dot2 = ivecs.at(v2*3)*ivecs.at(k*3) + ivecs.at(v2*3+1)*ivecs.at(k*3+1) + ivecs.at(v2*3+2)*ivecs.at(k*3+2);
      if (dot2 < cutoff2) {
        //found a suitable third vertex, store vertex 3 and break
        c++;
        indo[2] = idxs.at(i, k);
        dt[2] = 1.0 / pow(errs.at(i, k), 2);
        break;
      }
    }
    //all queries on one side, no interpolation
    if (c == 0) {
      for (ssize_t j = 0; j < dcols; j++){
        pout[row + j] = srclum.at(idxs.at(i, 0), j);
      }
    } else {
      nd = 0;
      for (ssize_t k = 0; k <= c; k++) {
        nd += dt[k];
      }
      for (ssize_t j = 0; j < dcols; j++){
        outv = 0;
        for (ssize_t k = 0; k <= c; k++) {
          outv += srclum.at(indo[k], j) * dt[k] / nd;
        }
        pout[row + j] = outv;
      }
    }
  }
  return arrout;
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
  m.def("interpolate_kdquery", &interpolate_kdquery,
        "destvec"_a,
        "errs"_a,
        "idxs"_a,
        "srcvec"_a,
        "srclum"_a,
        "err"_a=0.00436,
        interpolate_kdquery_docstring);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
