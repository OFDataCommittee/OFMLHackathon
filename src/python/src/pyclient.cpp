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


using namespace SmartRedis;

namespace py = pybind11;

PyClient::PyClient(bool cluster)
{
    Client* client = new Client(cluster);
    this->_client = client;
}

PyClient::~PyClient()
{
    delete this->_client;
}

void PyClient::put_tensor(std::string& key,
                          std::string& type,
                          py::array data)
{
    auto buffer = data.request();
    void* ptr = buffer.ptr;

    // get dims
    std::vector<size_t> dims(buffer.ndim);
    for (int i=0; i < buffer.shape.size(); i++) {
        dims[i] = (size_t) buffer.shape[i];
    }

    TensorType ttype = TENSOR_TYPE_MAP.at(type);

    try {
        this->_client->put_tensor(key, ptr, dims, ttype,
                                    MemoryLayout::contiguous);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return;
}

py::array PyClient::get_tensor(const std::string& key)
{
    TensorBase* tensor;
    try {
        tensor = this->_client->_get_tensorbase_obj(key);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }

    //Define py::capsule lambda function for destructor
    py::capsule free_when_done((void*)tensor, [](void *tensor) {
            delete (TensorBase*)tensor;
            });

    // detect data type
    switch(tensor->type()) {
        case TensorType::dbl : {
            double* data =
                (double*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        case TensorType::flt : {
            float* data =
                (float*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        case TensorType::int64 : {
            int64_t* data =
                (int64_t*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        case TensorType::int32: {
            int32_t* data =
                (int32_t*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        case TensorType::int16: {
            int16_t* data =
                (int16_t*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        case TensorType::int8: {
            int8_t* data =
                (int8_t*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        case TensorType::uint16: {
            uint16_t* data =
                (uint16_t*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        case TensorType::uint8: {
            uint8_t* data =
                (uint8_t*)(tensor->data_view(MemoryLayout::contiguous));
            return py::array(tensor->dims(), data, free_when_done);
            break;
        }
        default :
            throw std::runtime_error("Could not infer type in "\
                                     "PyClient::get_tensor().");
            break;
    }
}

void PyClient::delete_tensor(const std::string& key) {
    try {
        this->_client->delete_tensor(key);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return;
}

void PyClient::copy_tensor(const std::string& key,
                           const std::string& dest_key) {
    try {
        this->_client->copy_tensor(key, dest_key);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return;
}

void PyClient::rename_tensor(const std::string& key,
                             const std::string& new_key) {
    try {
        this->_client->rename_tensor(key, new_key);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return;
}

void PyClient::put_dataset(PyDataset& dataset)
{
    try {
        this->_client->put_dataset(*dataset.get());
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return;
}

PyDataset* PyClient::get_dataset(const std::string& name)
{
    DataSet* data;
    try {
        data = new DataSet(this->_client->get_dataset(name));
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    PyDataset* dataset = new PyDataset(*data);
    return dataset;
}

void PyClient::delete_dataset(const std::string& key) {
    try {
        this->_client->delete_dataset(key);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return;
}

void PyClient::copy_dataset(const std::string& key,
                           const std::string& dest_key) {
    try {
        this->_client->copy_dataset(key, dest_key);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return;
}

void PyClient::rename_dataset(const std::string& key,
                             const std::string& new_key) {
    try {
        this->_client->rename_dataset(key, new_key);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return;
}

void PyClient::set_script_from_file(const std::string& key,
                                    const std::string& device,
                                    const std::string& script_file)
{
    try {
        this->_client->set_script_from_file(key, device, script_file);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return;
}

void PyClient::set_script(const std::string& key,
                          const std::string& device,
                          const std::string_view& script)
{
    try {
        this->_client->set_script(key, device, script);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return;
}

std::string_view PyClient::get_script(const std::string& key)
{
    std::string_view script;
    try {
        script = this->_client->get_script(key);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return script;
}

void PyClient::run_script(const std::string& key,
                const std::string& function,
                std::vector<std::string>& inputs,
                std::vector<std::string>& outputs)
{
    try {
      this->_client->run_script(key, function, inputs, outputs);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return;
}

py::bytes PyClient::get_model(const std::string& key)
{
    std::string model;
    try {
        model = std::string(this->_client->get_model(key));
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    model = py::bytes(model);
    return model;
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
        this->_client->set_model(key, model, backend, device,
                                batch_size, min_batch_size, tag,
                                inputs, outputs);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return;
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
        this->_client->set_model_from_file(key, model_file, backend, device,
                                           batch_size, min_batch_size, tag,
                                           inputs, outputs);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return;
}

void PyClient::run_model(const std::string& key,
                         std::vector<std::string> inputs,
                         std::vector<std::string> outputs)
{
    try {
        this->_client->run_model(key, inputs, outputs);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return;
}

void PyClient::set_data_source(const std::string& source_id)
{
    this->_client->set_data_source(source_id);
    return;
}

bool PyClient::key_exists(const std::string& key)
{
    bool result;
    try {
        result = this->_client->key_exists(key);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return result;
}

bool PyClient::poll_key(const std::string& key,
                        int poll_frequency_ms,
                        int num_tries)
{
    bool result;
    try {
        result = this->_client->poll_key(key, poll_frequency_ms,
                                         num_tries);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return result;
}

bool PyClient::model_exists(const std::string& name)
{
    bool result;
    try {
        result = this->_client->model_exists(name);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return result;
}

bool PyClient::tensor_exists(const std::string& name)
{
  return this->_client->tensor_exists(name);
}

bool PyClient::poll_tensor(const std::string& name,
                           int poll_frequency_ms,
                           int num_tries)
{
    bool result;
    try {
        result = this->_client->poll_tensor(name, poll_frequency_ms,
                                            num_tries);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return result;
}

bool PyClient::poll_model(const std::string& name,
                          int poll_frequency_ms,
                          int num_tries)
{
    bool result;
    try {
        result = this->_client->poll_model(name, poll_frequency_ms,
                                           num_tries);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(e.what());
    }
    return result;
}

void PyClient::use_tensor_ensemble_prefix(bool use_prefix)
{
  this->_client->use_tensor_ensemble_prefix(use_prefix);
}

void PyClient::use_model_ensemble_prefix(bool use_prefix)
{
  this->_client->use_model_ensemble_prefix(use_prefix);
}

std::vector<py::dict> PyClient::get_db_node_info(std::vector<std::string> addresses)
{
    std::vector<py::dict> addresses_info;
    for(size_t i=0; i<addresses.size(); i++) {
        parsed_map info_map = this->_client->get_db_node_info(addresses[i]);
        py::dict info_dict = py::cast(info_map);
        addresses_info.push_back(info_dict);
    }
    return addresses_info;
}
