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
#include "tensorbase.h"
#include "tensor.h"
#include "srexception.h"

using namespace SmartRedis;

namespace py = pybind11;

PyClient::PyClient(bool cluster)
{
//    throw SRRuntimeException("Test");
    _client = NULL;
    try {
        _client = new Client(cluster);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "during client construction.");
    }
}

PyClient::~PyClient()
{
    if (_client != NULL) {
        delete _client;
        _client = NULL;
    }
}

void PyClient::put_tensor(std::string& name,
                          std::string& type,
                          py::array data)
{
    auto buffer = data.request();
    void* ptr = buffer.ptr;

    // get dims
    std::vector<size_t> dims(buffer.ndim);
    for (size_t i = 0; i < buffer.shape.size(); i++) {
        dims[i] = (size_t)buffer.shape[i];
    }

    SRTensorType ttype = TENSOR_TYPE_MAP.at(type);

    try {
        _client->put_tensor(name, ptr, dims, ttype, SRMemLayoutContiguous);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing put_tensor.");
    }
}

py::array PyClient::get_tensor(const std::string& name)
{
    TensorBase* tensor = NULL;
    try {
        tensor = _client->_get_tensorbase_obj(name);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing get_tensor.");
    }

    // Define py::capsule lambda function for destructor
    py::capsule free_when_done((void*)tensor, [](void *tensor) {
            delete reinterpret_cast<TensorBase*>(tensor);
            });

    // detect data type
    switch (tensor->type()) {
        case SRTensorTypeDouble: {
            double* data = reinterpret_cast<double*>(tensor->data_view(
                SRMemLayoutContiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        case SRTensorTypeFloat: {
            float* data = reinterpret_cast<float*>(tensor->data_view(
                SRMemLayoutContiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        case SRTensorTypeInt64: {
            int64_t* data = reinterpret_cast<int64_t*>(tensor->data_view(
                SRMemLayoutContiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        case SRTensorTypeInt32: {
            int32_t* data = reinterpret_cast<int32_t*>(tensor->data_view(
                SRMemLayoutContiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        case SRTensorTypeInt16: {
            int16_t* data = reinterpret_cast<int16_t*>(tensor->data_view(
                SRMemLayoutContiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        case SRTensorTypeInt8: {
            int8_t* data = reinterpret_cast<int8_t*>(tensor->data_view(
                SRMemLayoutContiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        case SRTensorTypeUint16: {
            uint16_t* data = reinterpret_cast<uint16_t*>(tensor->data_view(
                SRMemLayoutContiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        case SRTensorTypeUint8: {
            uint8_t* data = reinterpret_cast<uint8_t*>(tensor->data_view(
                SRMemLayoutContiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        default :
            throw SRRuntimeException("Could not infer type in "\
                                      "PyClient::get_tensor().");
    }
}

void PyClient::delete_tensor(const std::string& name) {
    try {
        _client->delete_tensor(name);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing delete_tensor.");
    }
}

void PyClient::copy_tensor(const std::string& src_name,
                           const std::string& dest_name) {
    try {
        _client->copy_tensor(src_name, dest_name);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing copy_tensor.");
    }
}

void PyClient::rename_tensor(const std::string& old_name,
                             const std::string& new_name) {
    try {
        _client->rename_tensor(old_name, new_name);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing rename_tensor.");
    }
}

void PyClient::put_dataset(PyDataset& dataset)
{
    try {
        _client->put_dataset(*dataset.get());
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing put_dataset.");
    }
}

PyDataset* PyClient::get_dataset(const std::string& name)
{
    DataSet* data;
    try {
        data = new DataSet(_client->get_dataset(name));
    }
    catch (const std::bad_alloc& e) {
        data = NULL;
        throw SRBadAllocException("DataSet");
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing get_dataset.");
    }
    PyDataset* dataset = new PyDataset(data);
    return dataset;
}

void PyClient::delete_dataset(const std::string& name) {
    try {
        _client->delete_dataset(name);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing delete_dataset.");
    }
}

void PyClient::copy_dataset(const std::string& src_name,
                           const std::string& dest_name) {
    try {
        _client->copy_dataset(src_name, dest_name);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing copy_dataset.");
    }
}

void PyClient::rename_dataset(const std::string& old_name,
                             const std::string& new_name) {
    try {
        _client->rename_dataset(old_name, new_name);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing rename_dataset.");
    }
}

void PyClient::set_script_from_file(const std::string& name,
                                    const std::string& device,
                                    const std::string& script_file)
{
    try {
        _client->set_script_from_file(name, device, script_file);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing set_script_from_file.");
    }
}

void PyClient::set_script(const std::string& name,
                          const std::string& device,
                          const std::string_view& script)
{
    try {
        _client->set_script(name, device, script);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing set_script.");
    }
}

std::string_view PyClient::get_script(const std::string& name)
{
    std::string_view script;
    try {
        script = _client->get_script(name);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing get_script.");
    }
    return script;
}

void PyClient::run_script(const std::string& name,
                const std::string& function,
                std::vector<std::string>& inputs,
                std::vector<std::string>& outputs)
{
    try {
      _client->run_script(name, function, inputs, outputs);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing run_script.");
    }
}

void PyClient::delete_script(const std::string& name)
{
    try {
        _client->delete_script(name);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing delete_script.");
    }
}

py::bytes PyClient::get_model(const std::string& name)
{
    try {
        std::string model(_client->get_model(name));
        return py::bytes(model);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing get_model.");
    }
}

void PyClient::set_model(const std::string& name,
                 const std::string_view& model,
                 const std::string& backend,
                 const std::string& device,
                 int batch_size,
                 int min_batch_size,
                 const std::string& tag,
                 const std::vector<std::string>& inputs,
                 const std::vector<std::string>& outputs)
{
    try {
        _client->set_model(name, model, backend, device,
                           batch_size, min_batch_size, tag,
                           inputs, outputs);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing set_model.");
    }
}

void PyClient::set_model_from_file(const std::string& name,
                                   const std::string& model_file,
                                   const std::string& backend,
                                   const std::string& device,
                                   int batch_size,
                                   int min_batch_size,
                                   const std::string& tag,
                                   const std::vector<std::string>& inputs,
                                   const std::vector<std::string>& outputs)
{
    try {
        _client->set_model_from_file(name, model_file, backend, device,
                                           batch_size, min_batch_size, tag,
                                           inputs, outputs);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing set_model_from_file.");
    }
}

void PyClient::run_model(const std::string& name,
                         std::vector<std::string> inputs,
                         std::vector<std::string> outputs)
{
    try {
        _client->run_model(name, inputs, outputs);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing run_model.");
    }
}

void PyClient::delete_model(const std::string& name)
{
    try {
        _client->delete_model(name);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing delete_model.");
    }
}

void PyClient::set_data_source(const std::string& source_id)
{
    _client->set_data_source(source_id);
}

bool PyClient::key_exists(const std::string& key)
{
    try {
        return _client->key_exists(key);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing key_exists.");
    }
}

bool PyClient::poll_key(const std::string& key,
                        int poll_frequency_ms,
                        int num_tries)
{
    try {
        return _client->poll_key(key, poll_frequency_ms, num_tries);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing poll_key.");
    }
}

bool PyClient::model_exists(const std::string& name)
{
    try {
        return _client->model_exists(name);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing model_exists.");
    }
}

bool PyClient::tensor_exists(const std::string& name)
{
    try {
        return _client->tensor_exists(name);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing tensor_exists.");
    }
}

bool PyClient::dataset_exists(const std::string& name)
{
  return this->_client->dataset_exists(name);
}

bool PyClient::poll_tensor(const std::string& name,
                           int poll_frequency_ms,
                           int num_tries)
{
    try {
        return _client->poll_tensor(name, poll_frequency_ms, num_tries);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing poll_tensor.");
    }
}

bool PyClient::poll_dataset(const std::string& name,
                            int poll_frequency_ms,
                            int num_tries)
{
    try {
        return _client->poll_dataset(name, poll_frequency_ms, num_tries);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing poll_dataset.");
    }
}

bool PyClient::poll_model(const std::string& name,
                          int poll_frequency_ms,
                          int num_tries)
{
    try {
        return _client->poll_model(name, poll_frequency_ms, num_tries);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing poll_model.");
    }
}

void PyClient::use_tensor_ensemble_prefix(bool use_prefix)
{
  _client->use_tensor_ensemble_prefix(use_prefix);
}

void PyClient::use_model_ensemble_prefix(bool use_prefix)
{
  _client->use_model_ensemble_prefix(use_prefix);
}

std::vector<py::dict> PyClient::get_db_node_info(std::vector<std::string> addresses)
{
    try {
        std::vector<py::dict> addresses_info;
        for (size_t i = 0; i < addresses.size(); i++) {
            parsed_reply_nested_map info_map = _client->get_db_node_info(addresses[i]);
            py::dict info_dict = py::cast(info_map);
            addresses_info.push_back(info_dict);
        }
        return addresses_info;
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                    "while executing get_db_node_info.");
    }
}

std::vector<py::dict> PyClient::get_db_cluster_info(std::vector<std::string> addresses)
{
    try {
        std::vector<py::dict> addresses_info;
        for (size_t i = 0; i < addresses.size(); i++) {
            parsed_reply_map info_map = _client->get_db_cluster_info(addresses[i]);
            py::dict info_dict = py::cast(info_map);
            addresses_info.push_back(info_dict);
        }
        return addresses_info;
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing get_db_cluster_info.");
    }
}

// Execute AI.INFO command
std::vector<py::dict>
PyClient::get_ai_info(const std::vector<std::string>& addresses,
                      const std::string& key,
                      const bool reset_stat)

{
    try {
        std::vector<py::dict> ai_info;
        for (size_t i = 0; i < addresses.size(); i++) {
                parsed_reply_map result =
                    _client->get_ai_info(addresses[i], key, reset_stat);
                py::dict result_dict = py::cast(result);
                ai_info.push_back(result_dict);

        }
        return ai_info;
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "during client get_ai_info() execution.");
    }
}

// Delete all keys of all existing databases
void PyClient::flush_db(std::vector<std::string> addresses)
{
    for (size_t i = 0; i < addresses.size(); i++) {
        try {
            _client->flush_db(addresses[i]);
        }
        catch (Exception& e) {
            // exception is already prepared for caller
            throw;
        }
        catch (std::exception& e) {
            // should never happen
            throw SRInternalException(e.what());
        }
        catch (...) {
            // should never happen
            throw SRInternalException("A non-standard exception was encountered "\
                                      "while executing flush_db.");
        }
    }
}

// Read the configuration parameters of a running server
py::dict PyClient::config_get(std::string expression, std::string address)
{
    try {
        std::unordered_map<std::string,std::string> result_map = _client->config_get(expression, address);
        return py::cast(result_map);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing config_get.");
    }
}

// Reconfigure the server
void PyClient::config_set(std::string config_param, std::string value, std::string address)
{
    try {
        _client->config_set(config_param, value, address);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing config_set.");
    }
}

void PyClient::save(std::vector<std::string> addresses)
{
    for (size_t address_index = 0; address_index < addresses.size(); address_index++) {
        try {
            _client->save(addresses[address_index]);
        }
        catch (Exception& e) {
            // exception is already prepared for caller
            throw;
        }
        catch (std::exception& e) {
            // should never happen
            throw SRInternalException(e.what());
        }
        catch (...) {
            // should never happen
            throw SRInternalException("A non-standard exception was encountered "\
                                      "while executing save.");
        }
    }
}

// EOF
