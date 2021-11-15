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
#include "tensorbase.h"
#include "tensor.h"
#include "srexception.h"

using namespace SmartRedis;

namespace py = pybind11;

PyClient::PyClient(bool cluster)
{
    _client = NULL;
    try {
        _client = new Client(cluster);
    }
    catch (smart_error& e) {
        // exception is already prepared for caller
        throw e;
    }
    catch (std::exception& e) {
        // should never happen
        throw e;
    }
    catch (...) {
        // should never happen
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "construction.");
    }
}

PyClient::~PyClient()
{
    if (_client != NULL) {
        delete _client;
        _client = NULL;
    }
}

void PyClient::put_tensor(std::string& key,
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
        _client->put_tensor(key, ptr, dims, ttype, sr_layout_contiguous);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "put_tensor execution.");
    }
}

py::array PyClient::get_tensor(const std::string& key)
{
    TensorBase* tensor = NULL;
    try {
        tensor = _client->_get_tensorbase_obj(key);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "get_tensor execution.");
    }

    //Define py::capsule lambda function for destructor
    py::capsule free_when_done((void*)tensor, [](void *tensor) {
            delete reinterpret_cast<TensorBase*>(tensor);
            });

    // detect data type
    switch (tensor->type()) {
        case sr_tensor_dbl: {
            double* data = reinterpret_cast<double*>(tensor->data_view(
                sr_layout_contiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        case sr_tensor_flt: {
            float* data = reinterpret_cast<float*>(tensor->data_view(
                sr_layout_contiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        case sr_tensor_int64: {
            int64_t* data = reinterpret_cast<int64_t*>(tensor->data_view(
                sr_layout_contiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        case sr_tensor_int32: {
            int32_t* data = reinterpret_cast<int32_t*>(tensor->data_view(
                sr_layout_contiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        case sr_tensor_int16: {
            int16_t* data = reinterpret_cast<int16_t*>(tensor->data_view(
                sr_layout_contiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        case sr_tensor_int8: {
            int8_t* data = reinterpret_cast<int8_t*>(tensor->data_view(
                sr_layout_contiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        case sr_tensor_uint16: {
            uint16_t* data = reinterpret_cast<uint16_t*>(tensor->data_view(
                sr_layout_contiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        case sr_tensor_uint8: {
            uint8_t* data = reinterpret_cast<uint8_t*>(tensor->data_view(
                sr_layout_contiguous));
            return py::array(tensor->dims(), data, free_when_done);
        }
        default :
            throw smart_runtime_error("Could not infer type in "\
                                      "PyClient::get_tensor().");
    }
}

void PyClient::delete_tensor(const std::string& key) {
    try {
        _client->delete_tensor(key);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "delete_tensor execution.");
    }
}

void PyClient::copy_tensor(const std::string& key,
                           const std::string& dest_key) {
    try {
        _client->copy_tensor(key, dest_key);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "copy_tensor execution.");
    }
}

void PyClient::rename_tensor(const std::string& key,
                             const std::string& new_key) {
    try {
        _client->rename_tensor(key, new_key);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "rename_tensor execution.");
    }
}

void PyClient::put_dataset(PyDataset& dataset)
{
    try {
        _client->put_dataset(*dataset.get());
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "put_dataset execution.");
    }
}

PyDataset* PyClient::get_dataset(const std::string& name)
{
    DataSet* data;
    try {
        data = new DataSet(_client->get_dataset(name));
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "get_dataset execution.");
    }
    PyDataset* dataset = new PyDataset(data);
    return dataset;
}

void PyClient::delete_dataset(const std::string& key) {
    try {
        _client->delete_dataset(key);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "delete_dataset execution.");
    }
}

void PyClient::copy_dataset(const std::string& key,
                           const std::string& dest_key) {
    try {
        _client->copy_dataset(key, dest_key);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "copy_dataset execution.");
    }
}

void PyClient::rename_dataset(const std::string& key,
                             const std::string& new_key) {
    try {
        _client->rename_dataset(key, new_key);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "rename_dataset execution.");
    }
}

void PyClient::set_script_from_file(const std::string& key,
                                    const std::string& device,
                                    const std::string& script_file)
{
    try {
        _client->set_script_from_file(key, device, script_file);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "set_script_from_file execution.");
    }
}

void PyClient::set_script(const std::string& key,
                          const std::string& device,
                          const std::string_view& script)
{
    try {
        _client->set_script(key, device, script);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "set_script execution.");
    }
}

std::string_view PyClient::get_script(const std::string& key)
{
    std::string_view script;
    try {
        script = _client->get_script(key);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "get_script execution.");
    }
    return script;
}

void PyClient::run_script(const std::string& key,
                const std::string& function,
                std::vector<std::string>& inputs,
                std::vector<std::string>& outputs)
{
    try {
      _client->run_script(key, function, inputs, outputs);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "run_script execution.");
    }
}

py::bytes PyClient::get_model(const std::string& key)
{
    std::string model;
    try {
        model = std::string(_client->get_model(key));
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "get_model execution.");
    }
    return py::bytes(model);
}

void PyClient::set_model(const std::string& key,
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
        _client->set_model(key, model, backend, device,
                                batch_size, min_batch_size, tag,
                                inputs, outputs);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
}

void PyClient::set_model_from_file(const std::string& key,
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
        _client->set_model_from_file(key, model_file, backend, device,
                                           batch_size, min_batch_size, tag,
                                           inputs, outputs);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "set_model_from_file execution.");
    }
}

void PyClient::run_model(const std::string& key,
                         std::vector<std::string> inputs,
                         std::vector<std::string> outputs)
{
    try {
        _client->run_model(key, inputs, outputs);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "run_model execution.");
    }
}

void PyClient::set_data_source(const std::string& source_id)
{
    _client->set_data_source(source_id);
}

bool PyClient::key_exists(const std::string& key)
{
    bool result = false;
    try {
        result = _client->key_exists(key);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "key_exists execution.");
    }
    return result;
}

bool PyClient::poll_key(const std::string& key,
                        int poll_frequency_ms,
                        int num_tries)
{
    bool result = false;
    try {
        result = _client->poll_key(key, poll_frequency_ms,
                                         num_tries);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "poll_key execution.");
    }
    return result;
}

bool PyClient::model_exists(const std::string& name)
{
    bool result = false;
    try {
        result = _client->model_exists(name);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "model_exists execution.");
    }
    return result;
}

bool PyClient::tensor_exists(const std::string& name)
{
    bool result = false;
    try {
        result = _client->tensor_exists(name);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "tensor_exists execution.");
    }
    return result;
}

bool PyClient::dataset_exists(const std::string& name)
{
  return this->_client->dataset_exists(name);
}

bool PyClient::poll_tensor(const std::string& name,
                           int poll_frequency_ms,
                           int num_tries)
{
    bool result = false;
    try {
        result = _client->poll_tensor(name, poll_frequency_ms,
                                            num_tries);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "poll_tensor execution.");
    }
    return result;
}

bool PyClient::poll_model(const std::string& name,
                          int poll_frequency_ms,
                          int num_tries)
{
    bool result = false;
    try {
        result = _client->poll_model(name, poll_frequency_ms,
                                           num_tries);
    }
    catch(const std::exception& e) {
        throw smart_runtime_error(e.what());
    }
    catch(...) {
        throw smart_runtime_error("A non-standard exception "\
                                  "was encountered during client "\
                                  "poll_model execution.");
    }
    return result;
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
    std::vector<py::dict> addresses_info;
    for(size_t i = 0; i < addresses.size(); i++) {
        try {
            parsed_reply_nested_map info_map = _client->get_db_node_info(addresses[i]);
            py::dict info_dict = py::cast(info_map);
            addresses_info.push_back(info_dict);
        }
        catch(const std::exception& e) {
            throw smart_runtime_error(e.what());
        }
        catch(...) {
            throw smart_runtime_error("A non-standard exception "\
                                      "was encountered during client "\
                                      "get_db_node_info execution.");
        }

    }
    return addresses_info;
}

std::vector<py::dict> PyClient::get_db_cluster_info(std::vector<std::string> addresses)
{
    std::vector<py::dict> addresses_info;
    for(size_t i = 0; i < addresses.size(); i++) {
        try {
            parsed_reply_map info_map = _client->get_db_cluster_info(addresses[i]);
            py::dict info_dict = py::cast(info_map);
            addresses_info.push_back(info_dict);
        }
        catch(const std::exception& e) {
            throw smart_runtime_error(e.what());
        }
        catch(...) {
            throw smart_runtime_error("A non-standard exception "\
                                      "was encountered during client "\
                                      "get_db_cluster_info execution.");
        }
    }
    return addresses_info;
}
