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
  ray::ray_pdone(1);
  ray::ambdone();
  ray::ray_done_pmap();
}

void Rtrace::initialize(pybind11::object pyargv11) {
  Renderer::initialize(pyargv11.ptr());
  nproc = ray::rtinit(argc, argv);

}

void Rtrace::initc(int argcount, char** argvector) {
  Renderer::initc(argcount, argvector);
  nproc = ray::rtinit(argc, argv);
}

void Rtrace::updateOSpec(char *vs, char of) {
  ray::setoutput2(vs, of);
}

void Rtrace::resetInstance() {
  resetRadiance();
  delete renderer;
  renderer = nullptr;
}

void Rtrace::loadscene(char *octname) {
  ray::rtrace_loadscene(octname);
  ray::rtrace_setup(nproc);
}

namespace py = pybind11;
using namespace pybind11::literals;
PYBIND11_MODULE(rtrace_c, m) {
  py::class_<Rtrace>(m, "cRtrace")
          .def("get_instance", &Rtrace::getInstance, py::return_value_policy::reference)
          .def("reset_instance", [](py::args& args){Rtrace::resetInstance();})
          .def("reset", [](py::args& args){Rtrace::resetRadiance();})
          .def("initialize", &Rtrace::initialize,
               R"pbdoc(pyargv11 (a sequence of strings) must be a member of calling
instance and persist for duration of program)pbdoc")
          .def("load_scene", &Rtrace::loadscene)
          .def("call", &Rtrace::call, py::call_guard<py::gil_scoped_release>())
          .def("update_ospec", &Rtrace::updateOSpec,
                  "vs"_a, "of"_a='z')
          .def_property_readonly_static("version", [](py::object) { return ray::VersionID; });

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}