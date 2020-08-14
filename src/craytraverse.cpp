#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>

namespace py = pybind11;
const char* docstring = R"pbdoc(raytraverse helper functions written in c++)pbdoc";

void init_pyhelpers(py::module &);
void init_radiance(py::module &);

PYBIND11_MODULE(craytraverse, m) {
  m.doc() = docstring;
  init_pyhelpers(m);
  init_radiance(m);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}


