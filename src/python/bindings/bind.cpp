/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2022, Hewlett Packard Enterprise
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
#include "srexception.h"

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
        .def("delete_script", &PyClient::delete_script)
        .def("set_model", &PyClient::set_model)
        .def("set_model_from_file", &PyClient::set_model_from_file)
        .def("get_model", &PyClient::get_model)
        .def("run_model", &PyClient::run_model)
        .def("delete_model", &PyClient::delete_model)
        .def("key_exists", &PyClient::key_exists)
        .def("poll_key", &PyClient::poll_key)
        .def("model_exists", &PyClient::model_exists)
        .def("tensor_exists", &PyClient::tensor_exists)
        .def("dataset_exists", &PyClient::dataset_exists)
        .def("poll_model", &PyClient::poll_model)
        .def("poll_tensor", &PyClient::poll_tensor)
        .def("poll_dataset", &PyClient::poll_dataset)
        .def("set_data_source", &PyClient::set_data_source)
        .def("use_tensor_ensemble_prefix", &PyClient::use_tensor_ensemble_prefix)
        .def("use_model_ensemble_prefix", &PyClient::use_model_ensemble_prefix)
        .def("get_db_node_info", &PyClient::get_db_node_info)
        .def("get_db_cluster_info", &PyClient::get_db_cluster_info)
        .def("get_ai_info", &PyClient::get_ai_info)
        .def("flush_db", &PyClient::flush_db)
        .def("config_set", &PyClient::config_set)
        .def("config_get", &PyClient::config_get)
        .def("save", &PyClient::save);

    // Python Dataset class
    py::class_<PyDataset>(m, "PyDataset")
        .def(py::init<std::string&>())
        .def("add_tensor", &PyDataset::add_tensor)
        .def("get_tensor", &PyDataset::get_tensor)
        .def("add_meta_scalar", &PyDataset::add_meta_scalar)
        .def("add_meta_string", &PyDataset::add_meta_string)
        .def("get_meta_scalars", &PyDataset::get_meta_scalars)
        .def("get_meta_strings", &PyDataset::get_meta_strings)
        .def("get_name", &PyDataset::get_name);

    // Python exception classes
    static py::exception<SmartRedis::Exception>         exception_handler(m,          "RedisReplyError");
    static py::exception<SmartRedis::RuntimeException>  runtime_exception_handler(m,  "RedisRuntimeError",  exception_handler.ptr());
    static py::exception<SmartRedis::BadAllocException> badalloc_exception_handler(m, "RedisBadAllocError", exception_handler.ptr());
    static py::exception<SmartRedis::DatabaseException> database_exception_handler(m, "RedisDatabaseError", exception_handler.ptr());
    static py::exception<SmartRedis::TimeoutException>  timeout_exception_handler(m,  "RedisTimeoutError",  exception_handler.ptr());
    static py::exception<SmartRedis::InternalException> internal_exception_handler(m, "RedisInternalError", exception_handler.ptr());
    static py::exception<SmartRedis::KeyException>      key_exception_handler(m,      "RedisKeyError",      exception_handler.ptr());

    // Translate SmartRedis Exception classes to python error classes
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        }
        // Parameter exceptions map to Python ValueError
        catch (const ParameterException& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        }

        // Type exceptions map to Python TypeError
        catch (const TypeException& e) {
            PyErr_SetString(PyExc_TypeError, e.what());
        }

        // Everything else maps to a custom override
        catch (const SmartRedis::RuntimeException& e) {
            runtime_exception_handler(e.what());
        }
        catch (const SmartRedis::BadAllocException& e) {
            badalloc_exception_handler(e.what());
        }
        catch (const SmartRedis::DatabaseException& e) {
            database_exception_handler(e.what());
        }
        catch (const SmartRedis::TimeoutException& e) {
            timeout_exception_handler(e.what());
        }
        catch (const SmartRedis::InternalException& e) {
            internal_exception_handler(e.what());
        }
        catch (const SmartRedis::KeyException& e) {
            key_exception_handler(e.what());
        }
        catch (const SmartRedis::Exception& e) {
            exception_handler(e.what());
        }
    });
}

