#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(craytraverse, m) {
    m.doc() = R"pbdoc(
           raytraverse helper functions written in c++
    )pbdoc";

    m.def("add", py::vectorize(add), R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

