
#include "client.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(silc, m) {
    m.doc() = "silc client"; // optional module docstring

    py::class_<SmartSimClient>(m, "SmartSimClient")
        .def(py::init<bool, bool>())
        .def("put_tensor", &SmartSimClient::put_tensor)
        .def("get_tensor", &SmartSimClient::get_tensor);

}

// put tensor
void put_tensor(
    SmartSimClient client,
    std::string& key,
    std::string& type,
    py::array_t<uint8_t> data,
    std::vector<int>& dims) {

        auto buffer = data.request()
        uint8_t *ptr = (uint8_t *) buffer.ptr;
        size_t N = buffer.shape[0]
    }