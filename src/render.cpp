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

#include "render.hh"
#include "iostream"



/* -------------------------------------------------------------------------- */

int Renderer::initialize(PyObject *arglist){
  //code snippet to convert python sequence of strings to char**
  // from: https://stackoverflow.com/questions/60067092/passing-a-list-of-strings
  // -from-python-to-c-through-pybind11/60068350#60068350
  if (PySequence_Check(arglist)) {
    Py_ssize_t sz = PySequence_Size(arglist);
    argc = (int) sz;
    argv = (char **) malloc(sz * sizeof(char *));

    for (Py_ssize_t i = 0; i < sz; ++i) {
      PyObject *item = PySequence_GetItem(arglist, i);
      //assumes python 3 string (unicode)
      argv[i] = (char *) PyUnicode_AsUTF8(item);
      Py_DECREF(item);
      if (!argv[i] || PyErr_Occurred()) {
        free(argv);
        argv = nullptr;
        break;
      }
    }
  }
  if (!argv) {
    if (!PyErr_Occurred())
      PyErr_SetString(PyExc_TypeError, "could not convert input to argv");
    throw pybind11::error_already_set();
  }
  //end snippet
  return 0;
}

void Renderer::loadscene(char* octname) {
  octree = octname;
}

py::array_t<double> Renderer::operator()(py::array_t<double, py::array::c_style> &vecs) {
  return py::array_t<double>();
}
