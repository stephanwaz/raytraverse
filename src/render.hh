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

#ifndef RAYTRAVERSE_RENDER_H
#define RAYTRAVERSE_RENDER_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class Renderer {

protected:
    Renderer() = default;

public:
    ~Renderer() = default;
    int initialize(PyObject* arglist);
    virtual py::array_t<double> operator()(py::array_t<double, py::array::c_style> &vecs);
    virtual void loadscene(char* octname);

protected:
    int nproc = 1;
    int argc = 0;
    char* octree;
    char** argv = nullptr;
};

#endif //RAYTRAVERSE_RENDER_H
