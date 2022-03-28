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
#include <fstream>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

class Renderer {

protected:
    Renderer() = default;

public:
    int rvc = 1;
    ~Renderer() = default;
    int py_initialize(PyObject* arglist);
    virtual int initialize(int iargc, char** iargv);
    virtual double* operator()(double* vecs, int rows);
    virtual py::array_t<double> py_call(py::array_t<double, py::array::c_style> &vecs);
    virtual void loadscene(char* octname);

protected:
    int nproc = 1;
    int argc = 0;
    char* octree;
    char** argv = nullptr;
};


inline int read_vecs(char* inp, std::vector<double> *values){
  int rowok;
  int rows;
  double row[6];
  std::ifstream source;                    // build a read-Stream
  source.open(inp);  // open data
  rows = 0;
  for(std::string line; std::getline(source, line); )   //read stream line by line
  {
    std::istringstream in(line);      //make a stream for the line itself
    rowok = 0;
    for (double & x : row) {
      in >> x;
      rowok += !in.fail();
    }
    if (rowok == 6){
      rows++;
      for (double & x : row)
        values->push_back(x);
    }
  }
  return rows;
}

inline void print_result(double* buff, int rows, int dpr){
  for (int i = 0; i < rows; i++){
    for (int j = 0; j < dpr; j++){
      std::cout << "\t" << buff[i * dpr + j];
    }
    std::cout << std::endl;
  }
}

#endif //RAYTRAVERSE_RENDER_H
