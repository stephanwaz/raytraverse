//
// Created by Stephen Wasilewski on 8/14/20.
//

#include "render.h"
#include "iostream"



/* -------------------------------------------------------------------------- */

void Renderer::initialize(PyObject *pyargv){
  //code snippet to convert python sequence of strings to char**
  // from: https://stackoverflow.com/questions/60067092/passing-a-list-of-strings
  // -from-python-to-c-through-pybind11/60068350#60068350
  if (PySequence_Check(pyargv)) {
    Py_ssize_t sz = PySequence_Size(pyargv);
    argc = (int) sz;
    argv = (char **) malloc(sz * sizeof(char *));

    for (Py_ssize_t i = 0; i < sz; ++i) {
      PyObject *item = PySequence_GetItem(pyargv, i);
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
}

void Renderer::initc(int argcount, char **argvector) {
  argc = argcount;
  argv = argvector;
}

void Renderer::call(char *fname) {}

