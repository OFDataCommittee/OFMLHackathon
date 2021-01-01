#include "pyclient.h"

namespace py = pybind11;


PYBIND11_MODULE(silcPy, m) {
    m.doc() = "silc client"; // optional module docstring

    // Python client bindings
    py::class_<PyClient>(m, "PyClient")
        .def(py::init<bool, bool>())
        .def("put_tensor", &PyClient::put_tensor)
        .def("get_tensor", &PyClient::get_tensor)
        .def("put_dataset", &PyClient::put_dataset)
        .def("get_dataset", &PyClient::get_dataset);

    // Python Dataset class
    py::class_<PyDataset>(m, "PyDataset")
        .def(py::init<std::string&>())
        .def("add_tensor", &PyDataset::add_tensor)
        .def("get_tensor", &PyDataset::get_tensor);
}

