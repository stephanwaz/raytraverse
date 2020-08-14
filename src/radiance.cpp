#include <pybind11/pybind11.h>
#include <cstring>
#include <cstdio>

namespace ray{
#include <ray.h>
#include "rtcall.h"
}

#include "rtrace.h"

namespace py = pybind11;

using namespace pybind11::literals;
void init_radiance(py::module &m) {
  m.def("version", [](){return ray::VersionID;});
  py::class_<Rtrace>(m, "Rtrace")
          .def(py::init<pybind11::object>())
          .def("call", &Rtrace::call);
}
