//
// Created by Stephen Wasilewski on 8/14/20.
//
#include <iostream>
#include "rtrace.h"

namespace ray{
#include <ray.h>
#include "rtinit.h"
}

void Rtrace::call(char *fname) {
  ray::rtrace_call(fname);
}

Renderer& Rtrace::getInstance() {
  if (not Renderer::renderer) {
    Renderer::renderer = new Rtrace;
  }
  return *renderer;
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

