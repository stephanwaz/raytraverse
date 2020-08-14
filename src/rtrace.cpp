//
// Created by Stephen Wasilewski on 8/14/20.
//

#include "rtrace.h"

namespace ray{
#include <ray.h>
#include "rtcall.h"
}

Rtrace::Rtrace(pybind11::object pyargv11) : Renderer(pyargv11.ptr()) {
  nproc = ray::callmain(argc, argv);
  free(argv);
}

void Rtrace::call(std::string fname) {
  char *cstr = &fname[0];
  ray::rtrace(cstr, nproc);
}