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

#include "pyclient.h"
#include "tensorbase.h"
#include "tensor.h"
#include "srexception.h"

using namespace SmartRedis;

namespace py = pybind11;

// Decorator to standardize exception handling in PyBind Client API methods
template <class T>
auto pb_client_api(T&& client_api_func, const char* name)
{
  // we create a closure below
  auto decorated =
  [name, client_api_func = std::forward<T>(client_api_func)](auto&&... args)
  {
    try {
      return client_api_func(std::forward<decltype(args)>(args)...);
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
        std::string msg(
            "A non-standard exception was encountered while executing ");
        msg += name;
        throw SRInternalException(msg);
    }
  };
  return decorated;
}

// Macro to invoke the decorator with a lambda function
#define MAKE_CLIENT_API(stuff)\
    pb_client_api([&] { stuff }, __func__)()

PyClient::PyClient(const std::string& logger_name)
    : PySRObject(logger_name)
{
    MAKE_CLIENT_API({
        _client = new Client(logger_name);
    });
}

PyClient::PyClient(
    PyConfigOptions& config_options,
    const std::string& logger_name)
    : PySRObject(logger_name)
{
    MAKE_CLIENT_API({
        ConfigOptions* co = config_options.get();
        _client = new Client(co, logger_name);
    });
}

PyClient::PyClient(bool cluster, const std::string& logger_name)
    : PySRObject(logger_name)
{
    MAKE_CLIENT_API({
        _client = new Client(cluster, logger_name);
    });
}

PyClient::~PyClient()
{
    MAKE_CLIENT_API({
        if (_client != NULL) {
            delete _client;
            _client = NULL;
        }
    });
}

void PyClient::put_tensor(
    std::string& name, std::string& type, py::array data)
{
    MAKE_CLIENT_API({
        auto buffer = data.request();
        void* ptr = buffer.ptr;

        // get dims
        std::vector<size_t> dims(buffer.ndim);
        for (size_t i = 0; i < buffer.shape.size(); i++) {
            dims[i] = (size_t)buffer.shape[i];
        }

        SRTensorType ttype = TENSOR_TYPE_MAP.at(type);

        _client->put_tensor(name, ptr, dims, ttype, SRMemLayoutContiguous);
    });
}

py::array PyClient::get_tensor(const std::string& name)
{
    return MAKE_CLIENT_API({
        TensorBase* tensor = _client->_get_tensorbase_obj(name);

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
    });
}

void PyClient::delete_tensor(const std::string& name)
{
    MAKE_CLIENT_API({
        _client->delete_tensor(name);
    });
}

void PyClient::copy_tensor(const std::string& src_name,
                           const std::string& dest_name)
{
    MAKE_CLIENT_API({
        _client->copy_tensor(src_name, dest_name);
    });
}

void PyClient::rename_tensor(const std::string& old_name,
                             const std::string& new_name)
{
    MAKE_CLIENT_API({
        _client->rename_tensor(old_name, new_name);
    });
}

void PyClient::put_dataset(PyDataset& dataset)
{
    MAKE_CLIENT_API({
        _client->put_dataset(*(dataset.get()));
    });
}

PyDataset* PyClient::get_dataset(const std::string& name)
{
    return MAKE_CLIENT_API({
        DataSet* data = new DataSet(_client->get_dataset(name));
        return new PyDataset(data);
    });
}

void PyClient::delete_dataset(const std::string& name)
{
    MAKE_CLIENT_API({
        _client->delete_dataset(name);
    });
}

void PyClient::copy_dataset(
    const std::string& src_name, const std::string& dest_name)
{
    MAKE_CLIENT_API({
        _client->copy_dataset(src_name, dest_name);
    });
}

void PyClient::rename_dataset(
    const std::string& old_name, const std::string& new_name)
{
    MAKE_CLIENT_API({
        _client->rename_dataset(old_name, new_name);
    });
}

void PyClient::set_script_from_file(const std::string& name,
                                    const std::string& device,
                                    const std::string& script_file)
{
    MAKE_CLIENT_API({
        _client->set_script_from_file(name, device, script_file);
    });
}

void PyClient::set_script_from_file_multigpu(const std::string& name,
                                             const std::string& script_file,
                                             int first_gpu,
                                             int num_gpus)
{
    MAKE_CLIENT_API({
        _client->set_script_from_file_multigpu(
            name, script_file, first_gpu, num_gpus);
    });
}

void PyClient::set_script(const std::string& name,
                          const std::string& device,
                          const std::string_view& script)
{
    MAKE_CLIENT_API({
        _client->set_script(name, device, script);
    });
}

void PyClient::set_script_multigpu(const std::string& name,
                                   const std::string_view& script,
                                   int first_gpu,
                                   int num_gpus)
{
    MAKE_CLIENT_API({
        _client->set_script_multigpu(name, script, first_gpu, num_gpus);
    });
}

std::string_view PyClient::get_script(const std::string& name)
{
    return MAKE_CLIENT_API({
        return _client->get_script(name);
    });
}

void PyClient::run_script(const std::string& name,
                const std::string& function,
                std::vector<std::string>& inputs,
                std::vector<std::string>& outputs)
{
    MAKE_CLIENT_API({
        _client->run_script(name, function, inputs, outputs);
    });
}

void PyClient::run_script_multigpu(const std::string& name,
                                   const std::string& function,
                                   std::vector<std::string>& inputs,
                                   std::vector<std::string>& outputs,
                                   int offset,
                                   int first_gpu,
                                   int num_gpus)
{
    MAKE_CLIENT_API({
        _client->run_script_multigpu(
            name, function, inputs, outputs, offset, first_gpu, num_gpus);
    });
}

void PyClient::delete_script(const std::string& name)
{
    MAKE_CLIENT_API({
        _client->delete_script(name);
    });
}

void PyClient::delete_script_multigpu(
    const std::string& name, int first_gpu, int num_gpus)
{
    MAKE_CLIENT_API({
        _client->delete_script_multigpu(name, first_gpu, num_gpus);
    });
}

py::bytes PyClient::get_model(const std::string& name)
{
    return MAKE_CLIENT_API({
        std::string model(_client->get_model(name));
        return py::bytes(model);
    });
}

void PyClient::set_model(const std::string& name,
                 const std::string_view& model,
                 const std::string& backend,
                 const std::string& device,
                 int batch_size,
                 int min_batch_size,
                 int min_batch_timeout,
                 const std::string& tag,
                 const std::vector<std::string>& inputs,
                 const std::vector<std::string>& outputs)
{
    MAKE_CLIENT_API({
        _client->set_model(name, model, backend, device,
                           batch_size, min_batch_size, min_batch_timeout,
                           tag, inputs, outputs);
    });
}

void PyClient::set_model_multigpu(const std::string& name,
                                  const std::string_view& model,
                                  const std::string& backend,
                                  int first_gpu,
                                  int num_gpus,
                                  int batch_size,
                                  int min_batch_size,
                                  int min_batch_timeout,
                                  const std::string& tag,
                                  const std::vector<std::string>& inputs,
                                  const std::vector<std::string>& outputs)
{
    MAKE_CLIENT_API({
        _client->set_model_multigpu(name, model, backend, first_gpu, num_gpus,
                                    batch_size, min_batch_size, min_batch_timeout,
                                    tag, inputs, outputs);
    });
}

void PyClient::set_model_from_file(const std::string& name,
                                   const std::string& model_file,
                                   const std::string& backend,
                                   const std::string& device,
                                   int batch_size,
                                   int min_batch_size,
                                   int min_batch_timeout,
                                   const std::string& tag,
                                   const std::vector<std::string>& inputs,
                                   const std::vector<std::string>& outputs)
{
    MAKE_CLIENT_API({
        _client->set_model_from_file(name, model_file, backend, device,
                                           batch_size, min_batch_size, min_batch_timeout,
                                           tag, inputs, outputs);
    });
}

void PyClient::set_model_from_file_multigpu(const std::string& name,
                                            const std::string& model_file,
                                            const std::string& backend,
                                            int first_gpu,
                                            int num_gpus,
                                            int batch_size,
                                            int min_batch_size,
                                            int min_batch_timeout,
                                            const std::string& tag,
                                            const std::vector<std::string>& inputs,
                                            const std::vector<std::string>& outputs)
{
    MAKE_CLIENT_API({
        _client->set_model_from_file_multigpu(
            name, model_file, backend, first_gpu, num_gpus, batch_size,
            min_batch_size, min_batch_timeout, tag, inputs, outputs);
    });
}

void PyClient::run_model(const std::string& name,
                         std::vector<std::string> inputs,
                         std::vector<std::string> outputs)
{
    MAKE_CLIENT_API({
        _client->run_model(name, inputs, outputs);
    });
}

void PyClient::run_model_multigpu(const std::string& name,
                                  std::vector<std::string> inputs,
                                  std::vector<std::string> outputs,
                                  int offset,
                                  int first_gpu,
                                  int num_gpus)
{
    MAKE_CLIENT_API({
        _client->run_model_multigpu(name, inputs, outputs, offset, first_gpu, num_gpus);
    });
}

void PyClient::delete_model(const std::string& name)
{
    MAKE_CLIENT_API({
        _client->delete_model(name);
    });
}

void PyClient::delete_model_multigpu(
    const std::string& name, int first_gpu, int num_gpus)
{
    MAKE_CLIENT_API({
        _client->delete_model_multigpu(name, first_gpu, num_gpus);
    });
}

void PyClient::set_data_source(const std::string& source_id)
{
    MAKE_CLIENT_API({
        _client->set_data_source(source_id);
    });
}

bool PyClient::key_exists(const std::string& key)
{
    return MAKE_CLIENT_API({
        return _client->key_exists(key);
    });
}

bool PyClient::poll_key(const std::string& key,
                        int poll_frequency_ms,
                        int num_tries)
{
    return MAKE_CLIENT_API({
        return _client->poll_key(key, poll_frequency_ms, num_tries);
    });
}

bool PyClient::model_exists(const std::string& name)
{
    return MAKE_CLIENT_API({
        return _client->model_exists(name);
    });
}

bool PyClient::tensor_exists(const std::string& name)
{
    return MAKE_CLIENT_API({
        return _client->tensor_exists(name);
    });
}

bool PyClient::dataset_exists(const std::string& name)
{
    return MAKE_CLIENT_API({
        return this->_client->dataset_exists(name);
    });
}

bool PyClient::poll_tensor(const std::string& name,
                           int poll_frequency_ms,
                           int num_tries)
{
    return MAKE_CLIENT_API({
        return _client->poll_tensor(name, poll_frequency_ms, num_tries);
    });
}

bool PyClient::poll_dataset(const std::string& name,
                            int poll_frequency_ms,
                            int num_tries)
{
    return MAKE_CLIENT_API({
        return _client->poll_dataset(name, poll_frequency_ms, num_tries);
    });
}

bool PyClient::poll_model(const std::string& name,
                          int poll_frequency_ms,
                          int num_tries)
{
    return MAKE_CLIENT_API({
        return _client->poll_model(name, poll_frequency_ms, num_tries);
    });
}

void PyClient::use_tensor_ensemble_prefix(bool use_prefix)
{
    MAKE_CLIENT_API({
        _client->use_tensor_ensemble_prefix(use_prefix);
    });
}

void PyClient::use_dataset_ensemble_prefix(bool use_prefix)
{
    MAKE_CLIENT_API({
        _client->use_dataset_ensemble_prefix(use_prefix);
    });
}

void PyClient::use_model_ensemble_prefix(bool use_prefix)
{
    MAKE_CLIENT_API({
        _client->use_model_ensemble_prefix(use_prefix);
    });
}

void PyClient::use_list_ensemble_prefix(bool use_prefix)
{
    MAKE_CLIENT_API({
        _client->use_list_ensemble_prefix(use_prefix);
    });
}


std::vector<py::dict> PyClient::get_db_node_info(std::vector<std::string> addresses)
{
    return MAKE_CLIENT_API({
        std::vector<py::dict> addresses_info;
        for (size_t i = 0; i < addresses.size(); i++) {
            parsed_reply_nested_map info_map = _client->get_db_node_info(addresses[i]);
            py::dict info_dict = py::cast(info_map);
            addresses_info.push_back(info_dict);
        }
        return addresses_info;
    });
}

std::vector<py::dict> PyClient::get_db_cluster_info(std::vector<std::string> addresses)
{
    return MAKE_CLIENT_API({
        std::vector<py::dict> addresses_info;
        for (size_t i = 0; i < addresses.size(); i++) {
            parsed_reply_map info_map = _client->get_db_cluster_info(addresses[i]);
            py::dict info_dict = py::cast(info_map);
            addresses_info.push_back(info_dict);
        }
        return addresses_info;
    });
}

// Execute AI.INFO command
std::vector<py::dict> PyClient::get_ai_info(
    const std::vector<std::string>& addresses,
    const std::string& key,
    const bool reset_stat)
{
    return MAKE_CLIENT_API({
        std::vector<py::dict> ai_info;
        for (size_t i = 0; i < addresses.size(); i++) {
            parsed_reply_map result =
                _client->get_ai_info(addresses[i], key, reset_stat);
            ai_info.push_back(py::cast(result));
        }
        return ai_info;
    });
}

// Delete all keys of all existing databases
void PyClient::flush_db(std::vector<std::string> addresses)
{
    for (size_t i = 0; i < addresses.size(); i++) {
        MAKE_CLIENT_API({
            _client->flush_db(addresses[i]);
        });
    }
}

// Read the configuration parameters of a running server
py::dict PyClient::config_get(std::string expression, std::string address)
{
    return MAKE_CLIENT_API({
        auto result_map = _client->config_get(expression, address);
        return py::cast(result_map);
    });
}

// Reconfigure the server
void PyClient::config_set(
    std::string config_param, std::string value, std::string address)
{
    MAKE_CLIENT_API({
        _client->config_set(config_param, value, address);
    });
}

// Save a copy of the database
void PyClient::save(std::vector<std::string> addresses)
{
    for (size_t address_index = 0; address_index < addresses.size(); address_index++) {
        MAKE_CLIENT_API({
            _client->save(addresses[address_index]);
        });
    }
}


// Appends a dataset to the aggregation list
void PyClient::append_to_list(const std::string& list_name, PyDataset& dataset)
{
    MAKE_CLIENT_API({
        _client->append_to_list(list_name, *dataset.get());
    });
}

// Delete an aggregation list
void PyClient::delete_list(const std::string& list_name)
{
    MAKE_CLIENT_API({
        _client->delete_list(list_name);
    });
}

// Copy an aggregation list
void PyClient::copy_list(const std::string& src_name, const std::string& dest_name)
{
    MAKE_CLIENT_API({
        _client->copy_list(src_name, dest_name);
    });
}

// Rename an aggregation list
void PyClient::rename_list(const std::string& src_name, const std::string& dest_name)
{
    MAKE_CLIENT_API({
        _client->rename_list(src_name, dest_name);
    });
}

// Get the number of entries in the list
int PyClient::get_list_length(const std::string& list_name)
{
    return MAKE_CLIENT_API({
        return _client->get_list_length(list_name);
    });
}

// Poll list length until length is equal
bool PyClient::poll_list_length(const std::string& name, int list_length,
                                int poll_frequency_ms, int num_tries)
{
    return MAKE_CLIENT_API({
        return _client->poll_list_length(
            name, list_length, poll_frequency_ms, num_tries);
    });
}

// Poll list length until length is greater than or equal
bool PyClient::poll_list_length_gte(const std::string& name, int list_length,
                                    int poll_frequency_ms, int num_tries)
{
    return MAKE_CLIENT_API({
        return _client->poll_list_length_gte(
            name, list_length, poll_frequency_ms, num_tries);
    });
}

// Poll list length until length is less than or equal
bool PyClient::poll_list_length_lte(const std::string& name, int list_length,
                                   int poll_frequency_ms, int num_tries)
{
    return MAKE_CLIENT_API({
        return _client->poll_list_length_lte(
            name, list_length, poll_frequency_ms, num_tries);
    });
}

// Get datasets from an aggregation list
py::list PyClient::get_datasets_from_list(const std::string& list_name)
{
    return MAKE_CLIENT_API({
        std::vector<DataSet> datasets = _client->get_datasets_from_list(list_name);
        std::vector<PyDataset*> result;
        for (auto it = datasets.begin(); it != datasets.end(); it++) {
            DataSet* ds = new DataSet(std::move(*it));
            result.push_back(new PyDataset(ds));
        }
        py::list result_list = py::cast(result);
        return result_list;
    });
}

// Get a range of datasets (by index) from an aggregation list
py::list PyClient::get_dataset_list_range(
    const std::string& list_name, const int start_index, const int end_index)
{
    return MAKE_CLIENT_API({
        std::vector<DataSet> datasets = _client->get_dataset_list_range(
            list_name, start_index, end_index);
        std::vector<PyDataset*> result;
        for (auto it = datasets.begin(); it != datasets.end(); it++) {
            DataSet* ds = new DataSet(std::move(*it));
            result.push_back(new PyDataset(ds));
        }
        py::list result_list = py::cast(result);
        return result_list;
    });
}

// Configure the Redis module chunk size
void PyClient::set_model_chunk_size(int chunk_size)
{
    return MAKE_CLIENT_API({
        return _client->set_model_chunk_size(chunk_size);
    });
}

// Create a string representation of the Client
std::string PyClient::to_string()
{
    return MAKE_CLIENT_API({
        return _client->to_string();
    });
}

// EOF
