//
// Created by Stephen Wasilewski on 8/14/20.
//
#include <iostream>
#include "rcontrib.hh"

namespace rayrc{
#include <ray.h>
#include "rcinit.h"
    extern "C" {
#include <ambient.h>
#include <pmapray.h>
    }
}

Rcontrib* Rcontrib::renderer = nullptr;

void Rcontrib::call(char *fname) {
  rayrc::rcontrib_call(fname);
//  rayrc::end_children(0);
}

Rcontrib& Rcontrib::getInstance() {
  if (not renderer) {
    renderer = new Rcontrib;
  }
  return *renderer;
}

void Rcontrib::initialize(pybind11::object pyargv11) {
  Renderer::initialize(pyargv11.ptr());
  nproc = rayrc::rcontrib_init(argc, argv);
}

void Rcontrib::initc(int argcount, char** argvector) {
  Renderer::initc(argcount, argvector);
  nproc = rayrc::rcontrib_init(argc, argv);
}

void Rcontrib::resetRadiance() {
  rayrc::ambsync();
  rayrc::ray_done_pmap();
  rayrc::ray_pdone(1);
  rayrc::rcontrib_clear();
}

void Rcontrib::resetInstance() {
  resetRadiance();
  delete renderer;
  renderer = nullptr;
}

namespace py = pybind11;
PYBIND11_MODULE(rcontrib_c, m) {
  py::class_<Rcontrib>(m, "cRcontrib")
          .def("get_instance", &Rcontrib::getInstance, py::return_value_policy::reference)
          .def("reset_instance", [](py::args& args){Rcontrib::resetInstance();})
          .def("reset", [](py::args& args){Rcontrib::resetRadiance();})
          .def("initialize", &Rcontrib::initialize)
          .def("call", &Rcontrib::call)
          .def_property_readonly_static("version", [](py::object) { return rayrc::VersionID; });

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}

