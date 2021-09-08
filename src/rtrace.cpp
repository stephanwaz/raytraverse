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
#include "rtrace.hh"

namespace ray{
#include "Radiance/src/rt/ray.h"
#include "csrc/rtinit.h"
    extern "C" {
#include "Radiance/src/rt/ambient.h"
#include "Radiance/src/rt/pmapray.h"
#include "Radiance/src/rt/func.h"
    }
}

Rtrace* Rtrace::renderer = nullptr;

py::array_t<double> Rtrace::operator()(py::array_t<double, py::array::c_style> &vecs) {
  int rows = vecs.shape(0);
  int cols = ray::return_value_count;
  py::buffer_info vbuff = vecs.request();
  auto *vptr = (double *) vbuff.ptr;

  double* buff = ray::rtrace_call(vptr, nproc, vecs.shape(0));

  return py::array_t<double>({rows, cols}, buff);

}

Rtrace& Rtrace::getInstance() {
  if (not Rtrace::renderer) {
    Rtrace::renderer = new Rtrace;
  }
  return *renderer;
}

void Rtrace::resetRadiance() {
  ray::ray_done(1);
  ray::dcleanup(2);
  ray::ray_restore(nullptr);
  srcobj = 0;
}

int Rtrace::initialize(pybind11::object arglist) {
  Renderer::initialize(arglist.ptr());
  ray::ray_restore(nullptr);
  nproc = ray::rtinit(argc, argv);
  return nproc;
}

int Rtrace::updateOSpec(char *vs) {
  return ray::setoutput2(vs);
}

void Rtrace::loadscene(char *octname) {
  Renderer::loadscene(octname);
  ray::ray_done(0);
  nproc = ray::rtinit(argc, argv);
  ray::rtrace_loadscene(octree);
  ray::rtrace_loadsrc(nullptr, 0);
}

void Rtrace::loadsrc(char *srcname, int freesrc) {
  if (freesrc < 0)
    freesrc = srcobj;
  srcobj += ray::rtrace_loadsrc(srcname, freesrc);
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

const char* doc_load_src =
        R"pbdoc(arglist (a sequence of strings) must be a member of calling
instance and persist for duration of program

updates private srcobj parameter for default removing all sources

Parameters
----------
srcname: str
    path to file with source definition.
freesrc: int, optional
    number of previous sources to unload (unloads from end of object list
    only safe if removing sources loaded by this function. If negative removes
    all sources loaded by this function.

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

const char* doc_update_ospec =
        R"pbdoc(update output values request

Parameters
----------
vs: str
    output specification string (see rtrace manpage option -o)

Returns
-------
ncomp: int
    number of components renderer will return, or -1 on failure.
)pbdoc";


PYBIND11_MODULE(rtrace_c, m){
  py::class_<Rtrace>(m, "cRtrace", py::module_local(), R"pbdoc(singleton interface to the Radiance rtrace executable.

See the rtrace man page for a full description of the programs functionality. Instance is initialized with a list
of arguments similar to the command line tool, but with several differences:

  - no -f format specifier, input and output is always a numpy array.
  - no -h option.
  - no -x/-y options, shape output data as necessary with np.reshape
  - no -P/-PP modes
  - -lr 0 behaves differently from radiance, sets a true reflection limit of 0 rather than disabling limit, for behavior
    approaching radiance, set -lr -1000
  - an additional -c N option repeats each input N times and averages the result. Make sure that uncorrelated sampling
    is used (-U+, default)
  - the default output is -oz, z is an additional output specifier that yields a single photopic brightness per input
    ray.
  - no s/m/M/t/T/~ allowed as output specifiers

Examples
--------

basic usage::

  from raytraverse.crenderer import cRtrace
  instance = cRtrace.get_instance()
  instance.initialize(["rtrace", ...]) #Note: do not include octree at end!
  instance.load_scene("scene.oct")
  # ...
  # define 'rays' as a numpy array of shape (N, 6)
  # ...
  lum = instance(rays)

cRtrace can also update the output specification and/or the settings without reloading the scene geometry::

  instance.update_ospec("L") # to query ray distance
  instance.initialize("rtrace -ab 0 -lr 0".split()) # note this begins with default arguments, it is not additive with previous settings!
  raylength = instance(rays)

but if you are loading new geometry, the instance should be reset::

  instance.reset()
  instance.initialize(["rtrace", ...])
  instance.load_scene("scene2.oct")

by loading a scene without light sources, sources can be dynamically loaded and unloaded without a reset::

  instance.reset()
  instance.initialize(["rtrace", ...])
  instance.load_scene("scene_no_sources.oct")
  instance.load_source("sky.rad")
  skylum = instance(rays)
  instance.load_source("sun.rad") # this unloads sky.rad and loads sun.rad
  sunlum = instance(rays)
  instance.load_source("sky.rad", 0) # using the optional freesrc, keep the sun loaded
  totallum = instance(rays)
  if np.allclose(skylum + sunlum, totallum, atol=.03): # depending on rendering settings / scene complexity
      print("light is additive!)

Notes
-----

the cRcontrib instance is best managed from a seperate class that handles argument generation.
See raytraverse.renderer.Rtrace

)pbdoc")
          .def("get_instance", [](){return Rtrace::getInstance();}, py::return_value_policy::reference, doc_get_instance)
          .def("reset", &Rtrace::resetRadiance, doc_reset)
          .def("initialize", &Rtrace::initialize, "arglist"_a, doc_initialize)
          .def("load_scene", &Rtrace::loadscene, "octree"_a, doc_load_scene)
          .def("load_source", &Rtrace::loadsrc, "srcname"_a, "freesrc"_a=-1, doc_load_src)
          .def("__call__", &Rtrace::operator(), "vecs"_a, doc_call)
          .def("update_ospec", &Rtrace::updateOSpec, "vs"_a, doc_update_ospec)
          .def_property_readonly_static("version", [](py::object) { return ray::VersionID; });

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
