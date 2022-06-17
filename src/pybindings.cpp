#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "gafact.cpp"

PYBIND11_MODULE(extensions, m) {
    py::class_<SolverSuppOutput>(m, "SolverSuppOutput")
        .def_readonly("rmse", &SolverSuppOutput::rmse)
        .def_readonly("objF", &SolverSuppOutput::objF)
        .def_readonly("niters", &SolverSuppOutput::niters);

    m.def("TAFact", &TAFact,
        "Solves D=T@A for T, A using non-negative matrix factorization\n"\
        "Expects D [n x m], T0 [k x m], A0[k x n]",
        py::return_value_policy::automatic,
        py::arg("D"), py::arg("Tt0"), py::arg("A0"),
        py::arg("lmbda")            = 0.0,  py::arg("itersMax") = 500,
        py::arg("innerItersMax")    = 500,  py::arg("tol")      = 1e-8,
        py::arg("tolA")             = 1e-7, py::arg("tolT")     = 1e-7,
        py::arg("nneg")             = true);
}
