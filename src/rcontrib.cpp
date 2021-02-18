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
#include <iostream>
#include "rcontrib.hh"

namespace rayrc{
#include <ray.h>
#include "csrc/rcinit.h"
    extern "C" {
#include <ambient.h>
#include <pmapray.h>
#include <func.h>
    }
}

namespace py = pybind11;

Rcontrib* Rcontrib::renderer = nullptr;

Rcontrib& Rcontrib::getInstance() {
  if (not renderer) {
    renderer = new Rcontrib;
  }
  return *renderer;
}

void Rcontrib::resetRadiance() {
  rayrc::ambsync();
  rayrc::ray_done_pmap();
  rayrc::ray_pdone(1);
  rayrc::rcontrib_clear();
  rayrc::dcleanup(2);
}

int Rcontrib::initialize(pybind11::object pyargv11) {
  Renderer::initialize(pyargv11.ptr());
  nproc = rayrc::rcontrib_init(argc, argv);
  return nproc;
}

void Rcontrib::initc(int argcount, char** argvector) {
  Renderer::initc(argcount, argvector);
  nproc = rayrc::rcontrib_init(argc, argv);
}

void Rcontrib::loadscene(char* octname) {
  Renderer::loadscene(octname);
  rayrc::ray_done(0);
  rayrc::rcontrib_loadscene(octree);
}

py::array_t<double> Rcontrib::operator()(py::array_t<double, py::array::c_style> &vecs) {
  int rows = vecs.shape(0);
  int cols = rayrc::return_value_count;
  py::buffer_info vbuff = vecs.request();
  auto *vptr = (double *) vbuff.ptr;
  rayrc::rcontrib_call(vptr, rows);
  double* buff = rayrc::output_values;

  return py::array_t<double>({rows, cols}, buff);
}

using namespace pybind11::literals;
PYBIND11_MODULE(rcontrib_c, m) {
  py::class_<Rcontrib>(m, "cRcontrib")
          .def("get_instance", &Rcontrib::getInstance, py::return_value_policy::reference)
          .def("reset", [](py::args& args){Rcontrib::resetRadiance();})
          .def("initialize", &Rcontrib::initialize)
          .def("load_scene", &Rcontrib::loadscene)
          .def("__call__", &Rcontrib::operator(), "vecs"_a)
          .def_property_readonly_static("version", [](py::object) { return rayrc::VersionID; });

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}

