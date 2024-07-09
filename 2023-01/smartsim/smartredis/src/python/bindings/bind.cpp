/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

#include "pysrobject.h"
#include "pyclient.h"
#include "pydataset.h"
#include "pylogcontext.h"
#include "srexception.h"
#include "pyconfigoptions.h"
#include "logger.h"

using namespace SmartRedis;
namespace py = pybind11;


PYBIND11_MODULE(smartredisPy, m) {
#define CLASS_METHOD(class, name) def(#name, &class::name)

    m.doc() = "smartredis client"; // optional module docstring

    // Python SRObject class
    #define SROBJECT_METHOD(name) CLASS_METHOD(PySRObject, name)
    py::class_<PySRObject>(m, "PySRObject")
        .def(py::init<std::string&>())
        .SROBJECT_METHOD(log_data)
        .SROBJECT_METHOD(log_warning)
        .SROBJECT_METHOD(log_error);

    // Python LogContext class
    py::class_<PyLogContext, PySRObject>(m, "PyLogContext")
        .def(py::init<std::string&>());

    // Python client class
    #define CLIENT_METHOD(name) CLASS_METHOD(PyClient, name)
    py::class_<PyClient, PySRObject>(m, "PyClient")
        .def(py::init<bool, const std::string&>())
        .def(py::init<const std::string&>())
        .def(py::init<PyConfigOptions&, const std::string&>())
        .CLIENT_METHOD(put_tensor)
        .CLIENT_METHOD(get_tensor)
        .CLIENT_METHOD(delete_tensor)
        .CLIENT_METHOD(copy_tensor)
        .CLIENT_METHOD(rename_tensor)
        .CLIENT_METHOD(put_dataset)
        .CLIENT_METHOD(get_dataset)
        .CLIENT_METHOD(delete_dataset)
        .CLIENT_METHOD(copy_dataset)
        .CLIENT_METHOD(rename_dataset)
        .CLIENT_METHOD(set_script_from_file)
        .CLIENT_METHOD(set_script_from_file_multigpu)
        .CLIENT_METHOD(set_script)
        .CLIENT_METHOD(set_script_multigpu)
        .CLIENT_METHOD(get_script)
        .CLIENT_METHOD(run_script)
        .CLIENT_METHOD(run_script_multigpu)
        .CLIENT_METHOD(delete_script)
        .CLIENT_METHOD(delete_script_multigpu)
        .CLIENT_METHOD(set_model)
        .CLIENT_METHOD(set_model_multigpu)
        .CLIENT_METHOD(set_model_from_file)
        .CLIENT_METHOD(set_model_from_file_multigpu)
        .CLIENT_METHOD(get_model)
        .CLIENT_METHOD(run_model)
        .CLIENT_METHOD(run_model_multigpu)
        .CLIENT_METHOD(delete_model)
        .CLIENT_METHOD(delete_model_multigpu)
        .CLIENT_METHOD(key_exists)
        .CLIENT_METHOD(poll_key)
        .CLIENT_METHOD(model_exists)
        .CLIENT_METHOD(tensor_exists)
        .CLIENT_METHOD(dataset_exists)
        .CLIENT_METHOD(poll_model)
        .CLIENT_METHOD(poll_tensor)
        .CLIENT_METHOD(poll_dataset)
        .CLIENT_METHOD(set_data_source)
        .CLIENT_METHOD(use_tensor_ensemble_prefix)
        .CLIENT_METHOD(use_dataset_ensemble_prefix)
        .CLIENT_METHOD(use_model_ensemble_prefix)
        .CLIENT_METHOD(use_list_ensemble_prefix)
        .CLIENT_METHOD(get_db_node_info)
        .CLIENT_METHOD(get_db_cluster_info)
        .CLIENT_METHOD(get_ai_info)
        .CLIENT_METHOD(flush_db)
        .CLIENT_METHOD(config_set)
        .CLIENT_METHOD(config_get)
        .CLIENT_METHOD(save)
        .CLIENT_METHOD(append_to_list)
        .CLIENT_METHOD(delete_list)
        .CLIENT_METHOD(copy_list)
        .CLIENT_METHOD(rename_list)
        .CLIENT_METHOD(get_list_length)
        .CLIENT_METHOD(poll_list_length)
        .CLIENT_METHOD(poll_list_length_gte)
        .CLIENT_METHOD(poll_list_length_lte)
        .CLIENT_METHOD(get_datasets_from_list)
        .CLIENT_METHOD(get_dataset_list_range)
        .CLIENT_METHOD(set_model_chunk_size)
        .CLIENT_METHOD(to_string)
    ;

    // Python Dataset class
    #define DATASET_METHOD(name) CLASS_METHOD(PyDataset, name)
    py::class_<PyDataset, PySRObject>(m, "PyDataset")
        .def(py::init<std::string&>())
        .DATASET_METHOD(add_tensor)
        .DATASET_METHOD(get_tensor)
        .DATASET_METHOD(add_meta_scalar)
        .DATASET_METHOD(add_meta_string)
        .DATASET_METHOD(get_meta_scalars)
        .DATASET_METHOD(get_meta_strings)
        .DATASET_METHOD(get_name)
        .DATASET_METHOD(get_metadata_field_names)
        .DATASET_METHOD(get_metadata_field_type)
        .DATASET_METHOD(get_tensor_type)
        .DATASET_METHOD(get_tensor_names)
        .DATASET_METHOD(get_tensor_dims)
        .DATASET_METHOD(to_string)
    ;

    // Python ConfigOptions class
    #define CONFIGOPTIONS_METHOD(name) CLASS_METHOD(PyConfigOptions, name)
    py::class_<PyConfigOptions>(m, "PyConfigOptions")
        .def_static("create_from_environment",
                    static_cast<PyConfigOptions* (*)(const std::string&)>(
                        &PyConfigOptions::create_from_environment))
        .CONFIGOPTIONS_METHOD(get_integer_option)
        .CONFIGOPTIONS_METHOD(get_string_option)
        .CONFIGOPTIONS_METHOD(is_configured)
        .CONFIGOPTIONS_METHOD(override_integer_option)
        .CONFIGOPTIONS_METHOD(override_string_option)
    ;

    // Logging functions
    m.def("cpp_log_data", py::overload_cast<const std::string&, SRLoggingLevel, const std::string&>(&log_data))
     .def("cpp_log_data", py::overload_cast<const SRObject*, SRLoggingLevel, const std::string&>(&log_data))
     .def("cpp_log_warning", py::overload_cast<const std::string&, SRLoggingLevel, const std::string&>(&log_warning))
     .def("cpp_log_warning", py::overload_cast<const SRObject*, SRLoggingLevel, const std::string&>(&log_warning))
     .def("cpp_log_error", py::overload_cast<const std::string&, SRLoggingLevel, const std::string&>(&log_error))
     .def("cpp_log_error", py::overload_cast<const SRObject*, SRLoggingLevel, const std::string&>(&log_error));

    // Logging levels
    py::enum_<SRLoggingLevel>(m, "SRLoggingLevel")
        .value("LLQuiet", LLQuiet)
        .value("LLInfo", LLInfo)
        .value("LLDebug", LLDebug)
        .value("LLDeveloper", LLDeveloper)
        .export_values();

    // Error management routines
    m.def("c_get_last_error_location", &SRGetLastErrorLocation);

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

