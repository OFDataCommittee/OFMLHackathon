#include "pyclient.h"
#include "pydataset.h"

namespace py = pybind11;


PYBIND11_MODULE(silcPy, m) {
    m.doc() = "silc client"; // optional module docstring

    // Python client bindings
    py::class_<SmartSimPyClient>(m, "Client")
        .def(py::init<bool, bool>())
        .def("put_tensor", &SmartSimPyClient::put_tensor)
        .def("get_tensor", &SmartSimPyClient::get_tensor);

    // Python Dataset class
    py::class_<PyDataset>(m, "Dataset")
        .def(py::init<std::string&>())
        .def("add_tensor", &PyDataset::add_tensor);
}

