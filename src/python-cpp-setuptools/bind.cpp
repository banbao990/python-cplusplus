#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

float add(float i, float j) {
    return i + j;
}

float multiply(float i, float j) {
    return i * j;
}

float divide(float i, float j = 1.0f) {
    return i / j;
}

namespace py = pybind11;

PYBIND11_MODULE(python_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: python_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
           multiply
           divide
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers
    )pbdoc");

    m.def(
        "subtract", [](float i, float j) { return i - j; }, R"pbdoc(
        Subtract two numbers
    )pbdoc");

    m.def("multiply", &multiply, R"pbdoc(
        Multiply two numbers
    )pbdoc",
          py::arg("i"), py::arg("j"));

    m.def("divide", &divide, R"pbdoc(
        Divide two numbers
    )pbdoc",
          py::arg("i"), py::arg("j") = 1.0f);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}