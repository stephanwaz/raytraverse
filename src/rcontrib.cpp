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
#include "Radiance/src/rt/ray.h"
#include "csrc/rcinit.h"
    extern "C" {
#include "Radiance/src/rt/ambient.h"
#include "Radiance/src/rt/pmapray.h"
#include "Radiance/src/rt/func.h"
    }
}

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

int Rcontrib::py_initialize(pybind11::object arglist) {
  Renderer::py_initialize(arglist.ptr());
//  for(int i = 0; i < argc; i++){
//    std::cout<<argv[i]<<std::endl;
//  }
  nproc = rayrc::rcontrib_init(argc, argv);
  return nproc;
}

int Rcontrib::initialize(int iargc, char **iargv) {
  Renderer::initialize(iargc, iargv);
  nproc = rayrc::rcontrib_init(argc, argv);
  rvc = rayrc::return_value_count;
  return nproc;
}

void Rcontrib::loadscene(char* octname) {
  Renderer::loadscene(octname);
  rayrc::ray_done(0);
  rayrc::rcontrib_loadscene(octree);
}

py::array_t<double> Rcontrib::py_call(py::array_t<double, py::array::c_style> &vecs) {
  int rows = vecs.shape(0);
  int cols = rayrc::return_value_count;
  py::buffer_info vbuff = vecs.request();
  auto *vptr = (double *) vbuff.ptr;
  rayrc::rcontrib_call(vptr, rows);
  double* buff = rayrc::output_values;
  return py::array_t<double>({rows, cols}, buff);
}

double* Rcontrib::operator()(double* vecs, int rows){
  rayrc::rcontrib_call(vecs, rows);
  double* buff = rayrc::output_values;
  return buff;
}

using namespace pybind11::literals;

const char* doc_get_instance = R"pbdoc(returns (instantiating if necessary) pointer to Renderer instance.)pbdoc";

const char* doc_reset =
        R"pbdoc(reset renderer state, must be called before loading an new scene or changing rendering
parameters)pbdoc";

const char* doc_initialize =
        R"pbdoc(arglist (a sequence of strings) must be a member of calling
instance and persist for duration of program

Parameters
----------
arglist: list
    a sequence of arguments to initialize renderer. must be a member of calling
    instance and persist for duration of program

Returns
-------
nproc: int
    number of processors renderer initialized with or -1 if initialization failed.
)pbdoc";

const char* doc_load_scene =
        R"pbdoc(load scene file to renderer

Parameters
----------
octee: str
    path to octree file.
)pbdoc";

const char* doc_call =
        R"pbdoc(run renderer for a set of rays

Parameters
----------
vecs: np.array
    shape (N, 6) origin + direction vectors

Returns
-------
values: np.array
    shape (N, M) result array, M depends on output specification
)pbdoc";


PYBIND11_MODULE(rcontrib_c, m) {
  py::class_<Rcontrib>(m, "cRcontrib", R"pbdoc(singleton interface to the Radiance rcontrib executable.

See the rcontrib man page for a full description of the programs functionality. Instance is initialized with a list
of arguments similar to the command line tool, but with several differences:

  - no -o option. All output is written to a memory buffer returned as a Numpy array
  - no -f format specifier, input and output is always a numpy array.
  - no -r option.
  - no -h option.
  - the -c option repeats and accumulates input rays rather than accumulating input.
  - an additional flag -Z outputs a single brightness value (photopic) rather than 3-color channels. this is True
    by default.

Examples
--------

basic usage::

  from raytraverse.crenderer import cRcontrib
  instance = cRcontrib.get_instance()
  instance.initialize(["rcontrib", "-n", "8", ..., "-m", "mod"])  #Note: do not include octree at end!
  instance.load_scene("scene.oct")
  # ...
  # define 'rays' as a numpy array of shape (N, 6)
  # ...
  contributions = instance(rays)

Subsequent calls can be made to the instance, but if either the settings or scene are changed::

  instance.reset()
  instance.initialize(["rcontrib", "-n", "8", ..., "-m", "mod2"])
  instance.load_scene("scene2.oct")

Notes
-----

the cRcontrib instance is best managed from a seperate class that handles argument generation.
See raytraverse.renderer.Rcontrib

)pbdoc")
          .def("get_instance", [](){return Rcontrib::getInstance();}, py::return_value_policy::reference, doc_get_instance)
          .def("reset", [](py::args& args){Rcontrib::resetRadiance();}, doc_reset)
          .def("initialize", &Rcontrib::py_initialize, "arglist"_a, doc_initialize)
          .def("load_scene", &Rcontrib::loadscene, "octree"_a, doc_load_scene)
          .def("__call__", &Rcontrib::py_call, "vecs"_a, doc_call)
          .def_property_readonly_static("version", [](py::object) { return rayrc::VersionID; });

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}

