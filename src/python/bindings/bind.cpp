#include "pyclient.h"

namespace py = pybind11;


PYBIND11_MODULE(silcPy, m) {
    m.doc() = "silc client"; // optional module docstring

    // Python client bindings
    py::class_<PyClient>(m, "PyClient")
        .def(py::init<bool>())
        .def("put_tensor", &PyClient::put_tensor)
        .def("get_tensor", &PyClient::get_tensor)
        .def("put_dataset", &PyClient::put_dataset)
        .def("get_dataset", &PyClient::get_dataset)
        .def("set_script_from_file", &PyClient::set_script_from_file)
        .def("set_script", &PyClient::set_script)
        .def("get_script", &PyClient::get_script)
        .def("run_script", &PyClient::run_script)
        .def("set_model", &PyClient::set_model)
        .def("set_model_from_file", &PyClient::set_model_from_file)
        .def("get_model", &PyClient::get_model)
        .def("run_model", &PyClient::run_model)
        .def("key_exists", &PyClient::key_exists)
        .def("poll_key", &PyClient::poll_key)
        .def("entity_exists", &PyClient::entity_exists)
        .def("poll_entity", &PyClient::poll_entity)
        .def("set_data_source", &PyClient::set_data_source)
        .def("use_ensemble_prefix", &PyClient::use_ensemble_prefix);
        //.def("build_tensor_key", &PyClient::build_tensor_key)
        //.def("build_model_key", &PyClient::build_model_key)
        //.def("build_dataset_meta_key", &PyClient::build_dataset_meta_key)
        //.def("build_dataset_tensor_key", &PyClient::build_dataset_tensor_key);

    // Python EntityType enum
    py::enum_<EntityType> (m, "EntityType")
        .value("tensor", EntityType::tensor)
        .value("dataset", EntityType::dataset)
        .value("model", EntityType::model)
        .export_values();

    // Python Dataset class
    py::class_<PyDataset>(m, "PyDataset")
        .def(py::init<std::string&>())
        .def("add_tensor", &PyDataset::add_tensor)
        .def("get_tensor", &PyDataset::get_tensor);
}

