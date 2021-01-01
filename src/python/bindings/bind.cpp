#include "pyclient.h"

namespace py = pybind11;


PYBIND11_MODULE(silcPy, m) {
    m.doc() = "silc client"; // optional module docstring

    // Python client bindings
    py::class_<SmartSimPyClient>(m, "Client")
        .def(py::init<bool, bool>())
        .def("put_tensor", &SmartSimPyClient::put_tensor)
        .def("get_tensor", &SmartSimPyClient::get_tensor)
        .def("put_dataset", &SmartSimPyClient::put_dataset)
        .def("get_dataset", &SmartSimPyClient::get_dataset);

    // Python Dataset class
    py::class_<PyDataset>(m, "Dataset")
        .def(py::init<std::string&>())
        .def("add_tensor", &PyDataset::add_tensor)
        .def("get_tensor", &PyDataset::get_tensor);
}

