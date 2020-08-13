#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

pybind11::tuple from_pdf(py::array_t<double> &pdf,
                         py::array_t<u_int32_t> candidates,
                         py::array_t<u_int32_t> bidx,
                         double threshold, double lb, double ub) {
  auto parr = pdf.unchecked<1>();
  auto carr = candidates.mutable_unchecked<1>();
  auto barr = bidx.mutable_unchecked<1>();

  u_int32_t ccnt = 0, bcnt = 0, nsampc = 0, cumsum = 0;
  bool cliplow, cliphigh, clip;
  double pdfnorm = 0;

  for (size_t idx = 0; idx < parr.shape(0); idx++) {
    cliplow = parr(idx) > threshold * lb;
    cliphigh = parr(idx) < (threshold * ub);
    clip = parr(idx) > threshold;
    // cliphigh
    if (parr(idx) < threshold * ub){
      // clip: set number of samples
      if (parr(idx) > threshold){
        nsampc++;
      }
      // cliplow: add to candidates and update probability
      if (parr(idx) > threshold * lb){
        carr(ccnt) = cumsum + ccnt;
//        pdfnorm += parr(idx);
//        parr(ccnt) = parr(idx);
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
  // normalize pdf
//  for (u_int32_t idx = 0; idx < ccnt; idx++) {
//    parr(idx) = parr(idx)/pdfnorm;
//  }
  return py::make_tuple(ccnt, bcnt, nsampc);
}

using namespace pybind11::literals;
PYBIND11_MODULE(craytraverse, m) {
    m.doc() = R"pbdoc(
           raytraverse helper functions written in c++
    )pbdoc";

    m.def("from_pdf", from_pdf,
            "pdf"_a,
            "candidates"_a,
            "bidx"_a,
            "threshold"_a,
            "lb"_a=.5,
            "ub"_a=4.0,
            R"pbdoc(helper function for draw.from_pdf

Parameters
----------
pdf: np.array
    array of doubles with unnormalized weights
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
    )pbdoc");



#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

