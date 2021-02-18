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

#ifndef RAYTRAVERSE_RTRACE_HH
#define RAYTRAVERSE_RTRACE_HH
#include "render.hh"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class Rtrace : public Renderer {

private:
    Rtrace() = default;
    static Rtrace* renderer;

public:
    static Rtrace& getInstance();
    int initialize(pybind11::object pyargv11);
    void loadscene(char* octname) override;
    void initc(int argc, char **argv) override;
    py::array_t<double> operator()(py::array_t<double, py::array::c_style> &vecs);
    int updateOSpec(char *vs);
    void resetRadiance();
    void loadsrc(char* srcname, int freesrc) override;
};


#endif //RAYTRAVERSE_RTRACE_HH
