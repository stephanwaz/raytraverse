//
// Created by Stephen Wasilewski on 8/14/20.
//
#include <iostream>
#include "rtrace.hh"

namespace ray{
#include <ray.h>
#include "rtinit.h"
    extern "C" {
#include <ambient.h>
#include <pmapray.h>
    }
}

Rtrace* Rtrace::renderer = nullptr;

void Rtrace::call(char *fname) {
  ray::rtrace_call(fname);
}

Rtrace& Rtrace::getInstance() {
  if (not Rtrace::renderer) {
    Rtrace::renderer = new Rtrace;
  }
  return *renderer;
}

void Rtrace::resetRadiance() {
  ray::ray_pdone(0);
  ray::ambdone();
  ray::ray_done_pmap();
}

void Rtrace::initialize(pybind11::object pyargv11) {
  Renderer::initialize(pyargv11.ptr());
  nproc = ray::rtinit(argc, argv);
  ray::rtrace_setup(nproc);
}

void Rtrace::initc(int argcount, char** argvector) {
  Renderer::initc(argcount, argvector);
  nproc = ray::rtinit(argc, argv);
  ray::rtrace_setup(nproc);
}

void Rtrace::updateOSpec(char *vs, char of) {
  ray::setoutput2(vs, of);
}

void Rtrace::resetInstance() {
  resetRadiance();
  delete renderer;
  renderer = nullptr;
}

namespace py = pybind11;
using namespace pybind11::literals;
PYBIND11_MODULE(rtrace_c, m) {
  py::class_<Rtrace>(m, "cRtrace")
          .def("get_instance", &Rtrace::getInstance, py::return_value_policy::reference)
          .def("reset_instance", [](py::args& args){Rtrace::resetInstance();})
          .def("reset", [](py::args& args){Rtrace::resetRadiance();})
          .def("initialize", &Rtrace::initialize)
          .def("call", &Rtrace::call)
          .def("update_ospec", &Rtrace::updateOSpec,
                  "vs"_a, "of"_a='z')
          .def_property_readonly_static("version", [](py::object) { return ray::VersionID; });

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}