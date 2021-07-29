/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Hewlett Packard Enterprise
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "pyclient.h"

using namespace SmartRedis;
namespace py = pybind11;


PYBIND11_MODULE(smartredisPy, m) {
    m.doc() = "smartredis client"; // optional module docstring

    // Python client bindings
    py::class_<PyClient>(m, "PyClient")
        .def(py::init<bool>())
        .def("put_tensor", &PyClient::put_tensor)
        .def("get_tensor", &PyClient::get_tensor)
        .def("delete_tensor", &PyClient::delete_tensor)
        .def("copy_tensor", &PyClient::copy_tensor)
        .def("rename_tensor", &PyClient::rename_tensor)
        .def("put_dataset", &PyClient::put_dataset)
        .def("get_dataset", &PyClient::get_dataset)
        .def("delete_dataset", &PyClient::delete_dataset)
        .def("copy_dataset", &PyClient::copy_dataset)
        .def("rename_dataset", &PyClient::rename_dataset)
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
        .def("model_exists", &PyClient::model_exists)
        .def("tensor_exists", &PyClient::tensor_exists)
        .def("poll_model", &PyClient::poll_model)
        .def("poll_tensor", &PyClient::poll_tensor)
        .def("set_data_source", &PyClient::set_data_source)
        .def("use_tensor_ensemble_prefix", &PyClient::use_tensor_ensemble_prefix)
        .def("use_model_ensemble_prefix", &PyClient::use_model_ensemble_prefix)
        .def("get_db_node_info", &PyClient::get_db_node_info)
        .def("get_db_cluster_info", &PyClient::get_db_cluster_info);

    // Python Dataset class
    py::class_<PyDataset>(m, "PyDataset")
        .def(py::init<std::string&>())
        .def("add_tensor", &PyDataset::add_tensor)
        .def("get_tensor", &PyDataset::get_tensor)
        .def("add_meta_scalar", &PyDataset::add_meta_scalar)
        .def("add_meta_string", &PyDataset::add_meta_string)
        .def("get_meta_scalars", &PyDataset::get_meta_scalars)
        .def("get_meta_strings", &PyDataset::get_meta_strings);
}

