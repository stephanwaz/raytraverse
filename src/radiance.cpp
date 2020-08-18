#include <pybind11/pybind11.h>
#include <cstring>
#include <cstdio>

namespace ray{
#include <ray.h>
#include "rtinit.h"
}

#include "rtrace.h"

namespace py = pybind11;

using namespace pybind11::literals;
void init_radiance(py::module &m) {
  m.def("version", [](){return ray::VersionID;});
  py::class_<Rtrace>(m, "Rtrace")
          .def("getInstance", &Rtrace::getInstance, py::return_value_policy::reference)
          .def("resetInstance", [](py::args& args){Rtrace::resetInstance();})
          .def("reset", [](py::args& args){Rtrace::resetRadiance();})
          .def("initialize", &Rtrace::initialize)
          .def("call", &Rtrace::call);
}
