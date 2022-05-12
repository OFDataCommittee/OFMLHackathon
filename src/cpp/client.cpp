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

#include <ctype.h>
#include "client.h"
#include "srexception.h"

using namespace SmartRedis;

// Constructor
Client::Client(bool cluster)
    : _redis_cluster(cluster ? new RedisCluster() : NULL),
      _redis(cluster ? NULL : new Redis())
{
    // A std::bad_alloc exception on the initializer will be caught
    // by the call to new for the client
    if (cluster)
        _redis_server =  _redis_cluster;
    else
        _redis_server =  _redis;
    _set_prefixes_from_env();
    _use_tensor_prefix = true;
    _use_model_prefix = false;
    _use_list_prefix = true;
}

// Destructor
Client::~Client()
{
    if (_redis_cluster != NULL)
    {
        delete _redis_cluster;
        _redis_cluster = NULL;
    }
    if (_redis != NULL)
    {
        delete _redis;
        _redis = NULL;
    }
    _redis_server = NULL;
}

// Put a DataSet object into the database
void Client::put_dataset(DataSet& dataset)
{
    CommandList cmds;
    _append_dataset_metadata_commands(cmds, dataset);
    _append_dataset_tensor_commands(cmds, dataset);
    _append_dataset_ack_command(cmds, dataset);
    _run(cmds);
}

// Retrieve a DataSet object from the database
DataSet Client::get_dataset(const std::string& name)
{
    // Get the metadata message and construct DataSet
    CommandReply reply = _get_dataset_metadata(name);

    // If the reply has no elements, it didn't exist
    if (reply.n_elements() == 0) {
        throw SRKeyException("The requested DataSet, \"" +
                             name + "\", does not exist.");
    }

    DataSet dataset(name);
    _unpack_dataset_metadata(dataset, reply);

    std::vector<std::string> tensor_names = dataset.get_tensor_names();

    // Retrieve DataSet tensors and fill the DataSet object
    for(size_t i = 0; i < tensor_names.size(); i++) {
        // Build the tensor key
        std::string tensor_key =
            _build_dataset_tensor_key(name, tensor_names[i], true);
        // Retrieve tensor and add it to the dataset
        _get_and_add_dataset_tensor(dataset, tensor_names[i], tensor_key);
    }

    return dataset;
}

// Rename the current dataset
void Client::rename_dataset(const std::string& old_name,
                            const std::string& new_name)
{
    copy_dataset(old_name, new_name);
    delete_dataset(old_name);
}

// Clone the dataset to a new name
void Client::copy_dataset(const std::string& src_name,
                          const std::string& dest_name)
{
    // Get the metadata message and construct DataSet
    CommandReply reply = _get_dataset_metadata(src_name);
    if (reply.n_elements() == 0) {
        throw SRKeyException("The requested DataSet " +
                             src_name + " does not exist.");
    }
    DataSet dataset(src_name);
    _unpack_dataset_metadata(dataset, reply);

    // Build tensor keys for cloning
    std::vector<std::string> tensor_names = dataset.get_tensor_names();
    std::vector<std::string> tensor_src_names =
        _build_dataset_tensor_keys(src_name, tensor_names, true);
    std::vector<std::string> tensor_dest_names =
         _build_dataset_tensor_keys(dest_name, tensor_names, false);

    // Clone tensors
    _redis_server->copy_tensors(tensor_src_names, tensor_dest_names);

    // Update the DataSet name to the destination name
    // so we can reuse the object for placing metadata
    // and ack commands
    dataset.set_name(dest_name);
    CommandList put_meta_cmds;
    _append_dataset_metadata_commands(put_meta_cmds, dataset);
    _append_dataset_ack_command(put_meta_cmds, dataset);
    (void)_run(put_meta_cmds);
}

// Delete a DataSet from the database.
// All tensors and metdata in the DataSet will be deleted.
void Client::delete_dataset(const std::string& name)
{
    CommandReply reply = _get_dataset_metadata(name);
    if (reply.n_elements() == 0) {
        throw SRRuntimeException("The requested DataSet " +
                                 name + " does not exist.");
    }

    DataSet dataset(name);
    _unpack_dataset_metadata(dataset, reply);

    // Delete the metadata (which contains the ack key)
    MultiKeyCommand cmd;
    cmd << "DEL" << Keyfield(_build_dataset_meta_key(dataset.get_name(), true));

    // Add in all the tensors to be deleted
    std::vector<std::string> tensor_names = dataset.get_tensor_names();
    std::vector<std::string> tensor_keys =
        _build_dataset_tensor_keys(dataset.get_name(), tensor_names, true);
    cmd.add_keys(tensor_keys);

    // Run the command
    reply = _run(cmd);

    if (reply.has_error()) {
        throw SRRuntimeException("An error was encountered when executing "\
                                 "DataSet " + name + " deletion.");
    }
}

// Put a tensor into the database
void Client::put_tensor(const std::string& name,
                        void* data,
                        const std::vector<size_t>& dims,
                        const SRTensorType type,
                        const SRMemoryLayout mem_layout)
{
    std::string key = _build_tensor_key(name, false);

    TensorBase* tensor = NULL;
    try {
        switch (type) {
            case SRTensorTypeDouble:
                tensor = new Tensor<double>(key, data, dims, type, mem_layout);
                break;
            case SRTensorTypeFloat:
                tensor = new Tensor<float>(key, data, dims, type, mem_layout);
                break;
            case SRTensorTypeInt64:
                tensor = new Tensor<int64_t>(key, data, dims, type, mem_layout);
                break;
            case SRTensorTypeInt32:
                tensor = new Tensor<int32_t>(key, data, dims, type, mem_layout);
                break;
            case SRTensorTypeInt16:
                tensor = new Tensor<int16_t>(key, data, dims, type, mem_layout);
                break;
            case SRTensorTypeInt8:
                tensor = new Tensor<int8_t>(key, data, dims, type, mem_layout);
                break;
            case SRTensorTypeUint16:
                tensor = new Tensor<uint16_t>(key, data, dims, type, mem_layout);
                break;
            case SRTensorTypeUint8:
                tensor = new Tensor<uint8_t>(key, data, dims, type, mem_layout);
                break;
            default:
                throw SRTypeException("Invalid type for put_tensor");
        }
    }
    catch (std::bad_alloc& e) {
        throw SRBadAllocException("tensor");
    }

    // Send the tensor
    CommandReply reply = _redis_server->put_tensor(*tensor);

    // Cleanup
    delete tensor;
    tensor = NULL;
    if (reply.has_error())
        throw SRRuntimeException("put_tensor failed");
}

// Get the tensor data, dimensions, and type for the provided tensor name.
// This function will allocate and retain management of the memory for the
// tensor data.
void Client::get_tensor(const std::string& name,
                        void*& data,
                        std::vector<size_t>& dims,
                        SRTensorType& type,
                        const SRMemoryLayout mem_layout)
{
    // Retrieve the TensorBase from the database
    TensorBase* ptr = _get_tensorbase_obj(name);

    // Set the user values
    dims = ptr->dims();
    type = ptr->type();
    data = ptr->data_view(mem_layout);

    // Hold the Tensor in memory for memory management
    _tensor_memory.add_tensor(ptr);
}

// Get the tensor data, dimensions, and type for the provided tensor name.
// This function will allocate and retain management of the memory for the
// tensor data and dimensions. This is a c-style interface for the tensor
// dimensions. Another function exists for std::vector dimensions.
 void Client::get_tensor(const std::string& name,
                        void*& data,
                        size_t*& dims,
                        size_t& n_dims,
                        SRTensorType& type,
                        const SRMemoryLayout mem_layout)
{

    std::vector<size_t> dims_vec;
    get_tensor(name, data, dims_vec, type, mem_layout);

    size_t dims_bytes = sizeof(size_t) * dims_vec.size();
    dims = _dim_queries.allocate_bytes(dims_bytes);
    n_dims = dims_vec.size();

    std::vector<size_t>::const_iterator it = dims_vec.cbegin();
    for (size_t i = 0; it != dims_vec.cend(); i++, it++)
        dims[i] = *it;
}

// Get tensor data and fill an already allocated array memory space that
// has the specified MemoryLayout. The provided type and dimensions are
// checked against retrieved values to ensure the provided memory space is
// sufficient. This method is the most memory efficient way to retrieve
// tensor data.
void Client::unpack_tensor(const std::string& name,
                           void* data,
                           const std::vector<size_t>& dims,
                           const SRTensorType type,
                           const SRMemoryLayout mem_layout)
{
    if (mem_layout == SRMemLayoutContiguous && dims.size() > 1) {
        throw SRRuntimeException("The destination memory space "\
                                 "dimension vector should only "\
                                 "be of size one if the memory "\
                                 "layout is contiguous.");
    }

    std::string get_key = _build_tensor_key(name, true);
    CommandReply reply = _redis_server->get_tensor(get_key);

    std::vector<size_t> reply_dims = GetTensorCommand::get_dims(reply);

    // Make sure we have the right dims to unpack into (Contiguous case)
    if (mem_layout == SRMemLayoutContiguous ||
        mem_layout == SRMemLayoutFortranContiguous) {
        size_t total_dims = 1;
        for (size_t i = 0; i < reply_dims.size(); i++) {
            total_dims *= reply_dims[i];
        }
        if (total_dims != dims[0] &&
            mem_layout == SRMemLayoutContiguous) {
            throw SRRuntimeException("The dimensions of the fetched "\
                                     "tensor do not match the length of "\
                                     "the contiguous memory space.");
        }
    }

    // Make sure we have the right dims to unpack into (Nested case)
    if (mem_layout == SRMemLayoutNested) {
        if (dims.size() != reply_dims.size()) {
            // Same number of dimensions
            throw SRRuntimeException("The number of dimensions of the  "\
                                     "fetched tensor, " +
                                     std::to_string(reply_dims.size()) + " "\
                                     "does not match the number of "\
                                     "dimensions of the user memory space, " +
                                     std::to_string(dims.size()));
        }

        // Same size in each dimension
        for (size_t i = 0; i < reply_dims.size(); i++) {
            if (dims[i] != reply_dims[i]) {
                throw SRRuntimeException("The dimensions of the fetched tensor "\
                                         "do not match the provided "\
                                         "dimensions of the user memory space.");
            }
        }
    }

    // Make sure we're unpacking the right type of data
    SRTensorType reply_type = GetTensorCommand::get_data_type(reply);
    if (type != reply_type)
        throw SRRuntimeException("The type of the fetched tensor "\
                                 "does not match the provided type");

    // Retrieve the tensor data into a Tensor
    std::string_view blob = GetTensorCommand::get_data_blob(reply);
    TensorBase* tensor = NULL;
    try {
        switch (reply_type) {
            case SRTensorTypeDouble:
                tensor = new Tensor<double>(get_key, (void*)blob.data(),
                                            reply_dims, reply_type,
                                            SRMemLayoutContiguous);
                break;
            case SRTensorTypeFloat:
                tensor = new Tensor<float>(get_key, (void*)blob.data(),
                                           reply_dims, reply_type,
                                           SRMemLayoutContiguous);
                break;
            case SRTensorTypeInt64:
                tensor = new Tensor<int64_t>(get_key, (void*)blob.data(),
                                            reply_dims, reply_type,
                                            SRMemLayoutContiguous);
                break;
            case SRTensorTypeInt32:
                tensor = new Tensor<int32_t>(get_key, (void*)blob.data(),
                                            reply_dims, reply_type,
                                            SRMemLayoutContiguous);
                break;
            case SRTensorTypeInt16:
                tensor = new Tensor<int16_t>(get_key, (void*)blob.data(),
                                            reply_dims, reply_type,
                                            SRMemLayoutContiguous);
                break;
            case SRTensorTypeInt8:
                tensor = new Tensor<int8_t>(get_key, (void*)blob.data(),
                                            reply_dims, reply_type,
                                            SRMemLayoutContiguous);
                break;
            case SRTensorTypeUint16:
                tensor = new Tensor<uint16_t>(get_key, (void*)blob.data(),
                                            reply_dims, reply_type,
                                            SRMemLayoutContiguous);
                break;
            case SRTensorTypeUint8:
                tensor = new Tensor<uint8_t>(get_key, (void*)blob.data(),
                                            reply_dims, reply_type,
                                            SRMemLayoutContiguous);
                break;
            default:
                throw SRTypeException("Invalid type for unpack_tensor");
        }
    }
    catch (std::bad_alloc& e) {
        throw SRBadAllocException("tensor");
    }

    // Unpack the tensor and reclaim it
    tensor->fill_mem_space(data, dims, mem_layout);
    delete tensor;
    tensor = NULL;
}

// Move a tensor from one name to another name
void Client::rename_tensor(const std::string& old_name,
                           const std::string& new_name)
{
    std::string old_key = _build_tensor_key(old_name, true);
    std::string new_key = _build_tensor_key(new_name, false);
    CommandReply reply = _redis_server->rename_tensor(old_key, new_key);
    if (reply.has_error())
        throw SRRuntimeException("rename_tensor failed");
}

// Delete a tensor from the database
void Client::delete_tensor(const std::string& name)
{
    std::string key = _build_tensor_key(name, true);
    CommandReply reply = _redis_server->delete_tensor(key);
    if (reply.has_error())
        throw SRRuntimeException("delete_tensor failed");
}

// Copy the tensor from the source name to the destination name
void Client::copy_tensor(const std::string& src_name,
                         const std::string& dest_name)
{
    std::string src_key = _build_tensor_key(src_name, true);
    std::string dest_key = _build_tensor_key(dest_name, false);
    CommandReply reply = _redis_server->copy_tensor(src_key, dest_key);
    if (reply.has_error())
        throw SRRuntimeException("copy_tensor failed");
}

// Set a model from file in the database for future execution
void Client::set_model_from_file(const std::string& name,
                                 const std::string& model_file,
                                 const std::string& backend,
                                 const std::string& device,
                                 int batch_size,
                                 int min_batch_size,
                                 const std::string& tag,
                                 const std::vector<std::string>& inputs,
                                 const std::vector<std::string>& outputs)
{
    if (model_file.size() == 0) {
        throw SRParameterException("model_file is a required "
                                   "parameter of set_model_from_file.");
    }

    std::ifstream fin(model_file, std::ios::binary);
    std::ostringstream ostream;
    ostream << fin.rdbuf();

    const std::string tmp = ostream.str();
    std::string_view model(tmp.data(), tmp.length());

    set_model(name, model, backend, device, batch_size,
              min_batch_size, tag, inputs, outputs);
}

// Set a model from file in the database for future execution in a multi-GPU system
void Client::set_model_from_file_multigpu(const std::string& name,
                                          const std::string& model_file,
                                          const std::string& backend,
                                          int first_gpu,
                                          int num_gpus,
                                          int batch_size,
                                          int min_batch_size,
                                          const std::string& tag,
                                          const std::vector<std::string>& inputs,
                                          const std::vector<std::string>& outputs)
{
    if (model_file.size() == 0) {
        throw SRParameterException("model_file is a required "
                                   "parameter of set_model_from_file_multigpu.");
    }

    std::ifstream fin(model_file, std::ios::binary);
    std::ostringstream ostream;
    ostream << fin.rdbuf();

    const std::string tmp = ostream.str();
    std::string_view model(tmp.data(), tmp.length());

    set_model_multigpu(name, model, backend, first_gpu, num_gpus, batch_size,
                       min_batch_size, tag, inputs, outputs);
}
// Set a model from a string buffer in the database for future execution
void Client::set_model(const std::string& name,
                       const std::string_view& model,
                       const std::string& backend,
                       const std::string& device,
                       int batch_size,
                       int min_batch_size,
                       const std::string& tag,
                       const std::vector<std::string>& inputs,
                       const std::vector<std::string>& outputs)
{
    if (name.size() == 0) {
        throw SRParameterException("name is a required parameter of set_model.");
    }

    if (backend.size() == 0) {
        throw SRParameterException("backend is a required "\
                                   "parameter of set_model.");
    }

    if (backend.compare("TF") != 0) {
        if (inputs.size() > 0) {
            throw SRParameterException("INPUTS in the model set command "\
                                       "is only valid for TF models");
        }
        if (outputs.size() > 0) {
            throw SRParameterException("OUTPUTS in the model set command "\
                                       "is only valid for TF models");
        }
    }

    const char* backends[] = { "TF", "TFLITE", "TORCH", "ONNX" };
    bool found = false;
    for (size_t i = 0; i < sizeof(backends)/sizeof(backends[0]); i++)
        found = found || (backend.compare(backends[i]) != 0);
    if (!found) {
        throw SRParameterException(backend + " is not a valid backend.");
    }

    if (device.size() == 0) {
        throw SRParameterException("device is a required "
                                   "parameter of set_model.");
    }
    if (device.compare("CPU") != 0 &&
        std::string(device).find("GPU") == std::string::npos) {
        throw SRRuntimeException(device + " is not a valid device.");
    }

    std::string key = _build_model_key(name, false);
    _redis_server->set_model(key, model, backend, device,
                             batch_size, min_batch_size,
                             tag, inputs, outputs);
}

void Client::set_model_multigpu(const std::string& name,
                                const std::string_view& model,
                                const std::string& backend,
                                int first_gpu,
                                int num_gpus,
                                int batch_size,
                                int min_batch_size,
                                const std::string& tag,
                                const std::vector<std::string>& inputs,
                                const std::vector<std::string>& outputs)
{
    if (name.size() == 0) {
        throw SRParameterException("name is a required parameter of set_model.");
    }
    if (backend.size() == 0) {
        throw SRParameterException("backend is a required "\
                                   "parameter of set_model.");
    }

    if (backend.compare("TF") != 0) {
        if (inputs.size() > 0) {
            throw SRParameterException("INPUTS in the model set command "\
                                       "is only valid for TF models");
        }
        if (outputs.size() > 0) {
            throw SRParameterException("OUTPUTS in the model set command "\
                                       "is only valid for TF models");
        }
    }

    if (first_gpu < 0) {
        throw SRParameterException("first_gpu must be a non-negative integer");
    }
    if (num_gpus < 1) {
        throw SRParameterException("num_gpus must be a positive integer.");
    }

    const char* backends[] = { "TF", "TFLITE", "TORCH", "ONNX" };
    bool found = false;
    for (size_t i = 0; i < sizeof(backends)/sizeof(backends[0]); i++)
        found = found || (backend.compare(backends[i]) != 0);
    if (!found) {
        throw SRParameterException(backend + " is not a valid backend.");
    }

    std::string key = _build_model_key(name, false);
    _redis_server->set_model_multigpu(
        key, model, backend, first_gpu, num_gpus,
        batch_size, min_batch_size,
        tag, inputs, outputs);
}


// Retrieve the model from the database
std::string_view Client::get_model(const std::string& name)
{
    std::string get_key = _build_model_key(name, true);
    CommandReply reply = _redis_server->get_model(get_key);
    if (reply.has_error())
        throw SRRuntimeException("failed to get model from server");

    char* model = _model_queries.allocate(reply.str_len());
    if (model == NULL)
        throw SRBadAllocException("model query");
    std::memcpy(model, reply.str(), reply.str_len());
    return std::string_view(model, reply.str_len());
}

// Set a script from file in the database for future execution
void Client::set_script_from_file(const std::string& name,
                                  const std::string& device,
                                  const std::string& script_file)
{
    // Read the script from the file
    std::ifstream fin(script_file);
    std::ostringstream ostream;
    ostream << fin.rdbuf();

    const std::string tmp = ostream.str();
    std::string_view script(tmp.data(), tmp.length());

    // Send it to the database
    set_script(name, device, script);
}

// Set a script from file in the database for future execution
// in a multi-GPU system
void Client::set_script_from_file_multigpu(const std::string& name,
                                           const std::string& script_file,
                                           int first_gpu,
                                           int num_gpus)
{
    // Read the script from the file
    std::ifstream fin(script_file);
    std::ostringstream ostream;
    ostream << fin.rdbuf();

    const std::string tmp = ostream.str();
    std::string_view script(tmp.data(), tmp.length());

    // Send it to the database
    set_script_multigpu(name, script, first_gpu, num_gpus);
}

// Set a script from a string buffer in the database for future execution
void Client::set_script(const std::string& name,
                        const std::string& device,
                        const std::string_view& script)
{
    if (device.size() == 0) {
        throw SRParameterException("device is a required "
                                   "parameter of set_script.");
    }
    if (device.compare("CPU") != 0 &&
        std::string(device).find("GPU") == std::string::npos) {
        throw SRRuntimeException(device + " is not a valid device.");
    }

    std::string key = _build_model_key(name, false);
    _redis_server->set_script(key, device, script);
}

// Set a script in the database for future execution in a multi-GPU system
void Client::set_script_multigpu(const std::string& name,
                                 const std::string_view& script,
                                 int first_gpu,
                                 int num_gpus)
{
    if (first_gpu < 0) {
        throw SRParameterException("first_gpu must be a non-negative integer.");
    }
    if (num_gpus < 1) {
        throw SRParameterException("num_gpus must be a positive integer.");
    }

    std::string key = _build_model_key(name, false);
    _redis_server->set_script_multigpu(key, script, first_gpu, num_gpus);
}

// Retrieve the script from the database
std::string_view Client::get_script(const std::string& name)
{
    std::string get_key = _build_model_key(name, true);
    CommandReply reply = _redis_server->get_script(get_key);
    char* script = _model_queries.allocate(reply.str_len());
    if (script == NULL)
        throw SRBadAllocException("model query");
    std::memcpy(script, reply.str(), reply.str_len());
    return std::string_view(script, reply.str_len());
}

// Run a model in the database using the specified input and output tensors
void Client::run_model(const std::string& name,
                       std::vector<std::string> inputs,
                       std::vector<std::string> outputs)
{
    std::string key = _build_model_key(name, true);

    if (_use_tensor_prefix) {
        _append_with_get_prefix(inputs);
        _append_with_put_prefix(outputs);
    }
    _redis_server->run_model(key, inputs, outputs);
}

// Run a model in the database using the
// specified input and output tensors in a multi-GPU system
void Client::run_model_multigpu(const std::string& name,
                                std::vector<std::string> inputs,
                                std::vector<std::string> outputs,
                                int offset,
                                int first_gpu,
                                int num_gpus)
{
    if (first_gpu < 0) {
        throw SRParameterException("first_gpu must be a non-negative integer.");
    }
    if (num_gpus < 1) {
        throw SRParameterException("num_gpus must be a positive integer.");
    }

    std::string key = _build_model_key(name, true);

    if (_use_tensor_prefix) {
        _append_with_get_prefix(inputs);
        _append_with_put_prefix(outputs);
    }
    _redis_server->run_model_multigpu(
        key, inputs, outputs, offset, first_gpu, num_gpus);
}

// Run a script function in the database using the specified input and output tensors
void Client::run_script(const std::string& name,
                        const std::string& function,
                        std::vector<std::string> inputs,
                        std::vector<std::string> outputs)
{
    std::string key = _build_model_key(name, true);

    if (_use_tensor_prefix) {
        _append_with_get_prefix(inputs);
        _append_with_put_prefix(outputs);
    }
    _redis_server->run_script(key, function, inputs, outputs);
}

// Run a script function in the database using the
// specified input and output tensors in a multi-GPU system
void Client::run_script_multigpu(const std::string& name,
                                 const std::string& function,
                                 std::vector<std::string> inputs,
                                 std::vector<std::string> outputs,
                                 int offset,
                                 int first_gpu,
                                 int num_gpus)
{
    if (first_gpu < 0) {
        throw SRParameterException("first_gpu must be a non-negative integer");
    }
    if (num_gpus < 1) {
        throw SRParameterException("num_gpus must be a positive integer.");
    }

    std::string key = _build_model_key(name, true);

    if (_use_tensor_prefix) {
        _append_with_get_prefix(inputs);
        _append_with_put_prefix(outputs);
    }
    _redis_server->run_script_multigpu(
        key, function, inputs, outputs, offset, first_gpu, num_gpus);
}

// Delete a model from the database
void Client::delete_model(const std::string& name)
{
    std::string key = _build_model_key(name, true);
    CommandReply reply = _redis_server->delete_model(key);

    if (reply.has_error())
        throw SRRuntimeException("AI.MODELDEL command failed on server");
}

// Delete a multiGPU model from the database
void Client::delete_model_multigpu(
    const std::string& name, int first_gpu, int num_gpus)
{
    if (first_gpu < 0) {
        throw SRParameterException("first_gpu must be a non-negative integer");
    }
    if (num_gpus < 1) {
        throw SRParameterException("num_gpus must be a positive integer.");
    }

    std::string key = _build_model_key(name, true);
    _redis_server->delete_model_multigpu(key, first_gpu, num_gpus);
}

// Delete a script from the database
void Client::delete_script(const std::string& name)
{
    std::string key = _build_model_key(name, true);
    CommandReply reply = _redis_server->delete_script(key);

    if (reply.has_error())
        throw SRRuntimeException("AI.SCRIPTDEL command failed on server");
}

// Delete a multiGPU script from the database
void Client::delete_script_multigpu(
    const std::string& name, int first_gpu, int num_gpus)
{
    if (first_gpu < 0) {
        throw SRParameterException("first_gpu must be a non-negative integer");
    }
    if (num_gpus < 1) {
        throw SRParameterException("num_gpus must be a positive integer.");
    }

    std::string key = _build_model_key(name, true);
    _redis_server->delete_script_multigpu(key, first_gpu, num_gpus);
}

// Check if the key exists in the database
bool Client::key_exists(const std::string& key)
{
    return _redis_server->key_exists(key);
}

// Check if the tensor (or the dataset) exists in the database
bool Client::tensor_exists(const std::string& name)
{
    std::string key = _build_tensor_key(name, true);
    return _redis_server->key_exists(key);
}

// Check if the dataset exists in the database
bool Client::dataset_exists(const std::string& name)
{
    std::string key = _build_dataset_ack_key(name, true);
    return _redis_server->hash_field_exists(key, _DATASET_ACK_FIELD);
}

// Check if the model (or the script) exists in the database
bool Client::model_exists(const std::string& name)
{
    std::string key = _build_model_key(name, true);
    return _redis_server->model_key_exists(key);
}

// Check if the key exists in the database at a specified frequency for a specified number of times
bool Client::poll_key(const std::string& key,
                      int poll_frequency_ms,
                      int num_tries)
{
    // Check for the key however many times requested
    for (int i = 0; i < num_tries; i++) {
        if (key_exists(key))
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_frequency_ms));
    }

    // If we get here, it was never found
    return false;
}

// Check if the model (or script) exists in the database at a specified frequency for a specified number of times.
bool Client::poll_model(const std::string& name,
                        int poll_frequency_ms,
                        int num_tries)
{
    // Check for the model/script however many times requested
    for (int i = 0; i < num_tries; i++) {
        if (model_exists(name))
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_frequency_ms));
    }

    // If we get here, it was never found
    return false;
}

// Check if the tensor exists in the database at a specified frequency for a specified number of times
bool Client::poll_tensor(const std::string& name,
                         int poll_frequency_ms,
                         int num_tries)
{
    // Check for the tensor however many times requested
    for (int i = 0; i < num_tries; i++) {
        if (tensor_exists(name))
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_frequency_ms));
    }

    // If we get here, it was never found
    return false;
}

// Check if the dataset exists in the database at a specified frequency for a specified number of times
bool Client::poll_dataset(const std::string& name,
                          int poll_frequency_ms,
                          int num_tries)
{
    // Check for the dataset however many times requested
    for (int i = 0; i < num_tries; i++) {
        if (dataset_exists(name))
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_frequency_ms));
    }

    // If we get here, it was never found
    return false;
}

// Establish a datasource
void Client::set_data_source(std::string source_id)
{
    // Validate the source prefix
    bool valid_prefix = false;
    size_t num_prefix = _get_key_prefixes.size();
    size_t save_index = -1;
    for (size_t i = 0; i < num_prefix; i++) {
        if (_get_key_prefixes[i].compare(source_id) == 0) {
            valid_prefix = true;
            save_index = i;
            break;
        }
    }

    if (!valid_prefix) {
        throw SRRuntimeException("Client error: data source " +
                                 std::string(source_id) +
                                 "could not be found during client "+
                                 "initialization.");
    }

    // Save the prefix
    _get_key_prefix = _get_key_prefixes[save_index];
}

// Set whether names of model and script entities should be prefixed
// (e.g. in an ensemble) to form database keys. Prefixes will only be
// used if they were previously set through the environment variables
// SSKEYOUT and SSKEYIN. Keys of entities created before this function
// is called will not be affected. By default, the client does not
// prefix model and script keys.
void Client::use_model_ensemble_prefix(bool use_prefix)
{
    _use_model_prefix = use_prefix;
}

// Set whether names of aggregation lists should be prefixed
// (e.g. in an ensemble) to form database keys. Prefixes will only be
// used if they were previously set through the environment variables
// SSKEYOUT and SSKEYIN. Keys of entities created before this function
// is called will not be affected. By default, the client prefixes
// aggregation list keys.
void Client::use_list_ensemble_prefix(bool use_prefix)
{
    _use_list_prefix = use_prefix;
}


// Set whether names of tensor and dataset entities should be prefixed
// (e.g. in an ensemble) to form database keys. Prefixes will only be used
// if they were previously set through the environment variables SSKEYOUT
// and SSKEYIN. Keys of entities created before this function is called
// will not be affected. By default, the client prefixes tensor and dataset
// keys with the first prefix specified with the SSKEYIN and SSKEYOUT
// environment variables.
void Client::use_tensor_ensemble_prefix(bool use_prefix)
{
    _use_tensor_prefix = use_prefix;
}

// Returns information about the given database node
parsed_reply_nested_map Client::get_db_node_info(std::string address)
{
    // Run an INFO EVERYTHING command to get node info
    DBInfoCommand cmd;
    std::string host = cmd.parse_host(address);
    uint64_t port = cmd.parse_port(address);

    cmd.set_exec_address_port(host, port);
    cmd << "INFO" << "EVERYTHING";
    CommandReply reply = _run(cmd);
    if (reply.has_error())
        throw SRRuntimeException("INFO EVERYTHING command failed on server");

    // Parse the results
    std::string db_node_info(reply.str(), reply.str_len());
    return DBInfoCommand::parse_db_node_info(db_node_info);
}

// Returns the CLUSTER INFO command reply addressed to a single cluster node.
parsed_reply_map Client::get_db_cluster_info(std::string address)
{
    if (_redis_cluster == NULL)
        throw SRRuntimeException("Cannot run on non-cluster environment");

    // Run the CLUSTER INFO command
    ClusterInfoCommand cmd;
    std::string host = cmd.parse_host(address);
    uint64_t port = cmd.parse_port(address);

    cmd.set_exec_address_port(host, port);
    cmd << "CLUSTER" << "INFO";
    CommandReply reply = _run(cmd);
    if (reply.has_error())
        throw SRRuntimeException("CLUSTER INFO command failed on server");

    // Parse the results
    std::string db_cluster_info(reply.str(), reply.str_len());
    return ClusterInfoCommand::parse_db_cluster_info(db_cluster_info);
}

// Returns the AI.INFO command reply
parsed_reply_map Client::get_ai_info(const std::string& address,
                                     const std::string& key,
                                     const bool reset_stat)
{
    // Run the command
    CommandReply reply =
        _redis_server->get_model_script_ai_info(address, key, reset_stat);

    if (reply.has_error())
        throw SRRuntimeException("AI.INFO command failed on server");

    if (reply.n_elements() % 2 != 0)
        throw SRInternalException("The AI.INFO reply structure has an "\
                                  "unexpected format");

    // Parse reply
    parsed_reply_map reply_map;
    for (size_t i = 0; i < reply.n_elements(); i += 2) {
        std::string map_key = reply[i].str();
        std::string value;
        if (reply[i + 1].redis_reply_type() == "REDIS_REPLY_STRING") {
            // Strip off a prefix if present. Form is {xx}.restofstring
            value = std::string(reply[i + 1].str(), reply[i + 1].str_len());
            if (_redis_cluster != NULL && value.size() > 0 && value[0] == '{') {
                size_t pos = value.find_first_of('}');
                if (pos != std::string::npos && pos + 2 < value.size() && value[pos + 1] == '.') {
                    value = value.substr(pos + 2, value.size() - (pos + 1));
                }
            }
        }
        else if (reply[i + 1].redis_reply_type() == "REDIS_REPLY_INTEGER")
            value = std::to_string(reply[i + 1].integer());
        else if (reply[i + 1].redis_reply_type() == "REDIS_REPLY_DOUBLE")
            value = std::to_string(reply[i + 1].dbl());
        else
            throw SRInternalException("The AI.INFO field " + map_key +
                                      " has an unexpected type.");
        reply_map[map_key] = value;
    }
    return reply_map;
}

// Delete all the keys of the given database
void Client::flush_db(std::string address)
{
    AddressAtCommand cmd;
    std::string host = cmd.parse_host(address);
    uint64_t port = cmd.parse_port(address);
    if (host.empty() or port == 0){
        throw SRRuntimeException(std::string(address) +
                                 "is not a valid database node address.");
    }
    cmd.set_exec_address_port(host, port);
    cmd << "FLUSHDB";

    CommandReply reply = _run(cmd);
    if (reply.has_error() > 0)
        throw SRRuntimeException("FLUSHDB command failed");
}

// Read the configuration parameters of a running server
std::unordered_map<std::string,std::string>
Client::config_get(std::string expression, std::string address)
{
    AddressAtCommand cmd;
    std::string host = cmd.parse_host(address);
    uint64_t port = cmd.parse_port(address);

    cmd.set_exec_address_port(host, port);
    cmd << "CONFIG" << "GET" << expression;

    CommandReply reply = _run(cmd);
    if (reply.has_error() > 0)
        throw SRRuntimeException("CONFIG GET command failed");

    // parse reply
    size_t n_dims = reply.n_elements();
    std::unordered_map<std::string,std::string> reply_map;
    for(size_t i = 0; i < n_dims; i += 2){
        reply_map[reply[i].str()] = reply[i+1].str();
    }

    return reply_map;
}

// Reconfigure the server
void Client::config_set(std::string config_param, std::string value, std::string address)
{
    AddressAtCommand cmd;
    std::string host = cmd.parse_host(address);
    uint64_t port = cmd.parse_port(address);

    cmd.set_exec_address_port(host, port);
    cmd << "CONFIG" << "SET" << config_param << value;

    CommandReply reply = _run(cmd);
    if (reply.has_error() > 0)
        throw SRRuntimeException("CONFIG SET command failed");
}

void Client::save(std::string address)
{
    AddressAtCommand cmd;
    std::string host = cmd.parse_host(address);
    uint64_t port = cmd.parse_port(address);

    cmd.set_exec_address_port(host, port);
    cmd << "SAVE";

    CommandReply reply = _run(cmd);
    if (reply.has_error() > 0)
        throw SRRuntimeException("SAVE command failed");
}

// Append dataset to aggregation list
void Client::append_to_list(const std::string& list_name,
                            const DataSet& dataset)
{
    // Build the list key
    std::string list_key = _build_list_key(list_name, false);

    // The aggregation list stores dataset key (not the meta key)
    std::string dataset_key = _build_dataset_key(dataset.get_name(), false);

    // Build the command
    SingleKeyCommand cmd;
    cmd << "RPUSH" << Keyfield(list_key) << dataset_key;

    // Run the command
    CommandReply reply = _run(cmd);
    if (reply.has_error() > 0)
        throw SRRuntimeException("RPUSH command failed. DataSet could not "\
                                 "be added to the aggregation list.");
}

// Delete an aggregation list
void Client::delete_list(const std::string& list_name)
{
    // Build the list key
    std::string list_key = _build_list_key(list_name, true);

    // Build the command
    SingleKeyCommand cmd;
    cmd << "DEL" << Keyfield(list_key);

    // Run the command
    CommandReply reply = _run(cmd);
    if (reply.has_error() > 0)
        throw SRRuntimeException("DEL command failed.");
}

// Copy aggregation list
void Client::copy_list(const std::string& src_name,
                       const std::string& dest_name)
{
    // Check for empty string inputs
    if (src_name.size() == 0) {
        throw SRParameterException("The src_name parameter cannot "\
                                   "be an empty string.");
    }

    if (dest_name.size() == 0) {
        throw SRParameterException("The dest_name parameter cannot "\
                                   "be an empty string.");
    }

    // If the source and destination names are the same, don't execute
    // any commands
    if (src_name == dest_name) {
        return;
    }

    // Build the source list key
    std::string src_list_key = _build_list_key(src_name, true);

    // Build the command to retrieve source list contents
    SingleKeyCommand cmd;
    cmd << "LRANGE" << Keyfield(src_list_key);
    cmd << std::to_string(0);
    cmd << std::to_string(-1);

    // Run the command to retrive the list contents
    CommandReply reply = _run(cmd);

    // Check for reply errors and correct type
    if (reply.has_error() > 0)
        throw SRRuntimeException("GET command failed. The aggregation "\
                                 "list could not be retrieved.");

    if (reply.redis_reply_type() != "REDIS_REPLY_ARRAY")
        throw SRRuntimeException("An unexpected type was returned for "
                                 "for the aggregation list.");

    if (reply.n_elements() == 0)
        throw SRRuntimeException("The source aggregation list does "\
                                 "not exist.");

    // Delete the current contents of the destination list or it will
    // be an append to the destination (not act as a renaming
    // of the original list)
    delete_list(dest_name);

    // Build the destination list key
    std::string dest_list_key = _build_list_key(dest_name, false);

    // The aggregation list contents will be directly added to a
    // Command using Command::add_field_ptr().  This means that
    // the above CommandReply must stay in scope and not destroy
    // it's memory.
    SingleKeyCommand copy_cmd;
    copy_cmd << "RPUSH" << Keyfield(dest_list_key);

    for (size_t i = 0; i < reply.n_elements(); i++) {
        // Check that the ith entry is a string (i.e. key)
        if (reply[i].redis_reply_type() != "REDIS_REPLY_STRING") {
            throw SRRuntimeException("Element " + std::to_string(i) +
                                     " in the aggregation list has an "\
                                     "unexpected type: " +
                                     reply.redis_reply_type());
        }

        // Check that the string length is not zero
        if(reply[i].str_len() == 0) {
            throw SRRuntimeException("Element " + std::to_string(i) +
                                     " contains an empty key, which is "\
                                     "not permitted.");
        }

        copy_cmd.add_field_ptr(reply[i].str(), reply[i].str_len());
    }

    CommandReply copy_reply = this->_run(copy_cmd);

    if (reply.has_error() > 0)
        throw SRRuntimeException("Dataset aggregation list copy "
                                 "operation failed.");
}

// Rename an aggregation list
void Client::rename_list(const std::string& src_name,
                         const std::string& dest_name)
{
    if (src_name.size() == 0) {
        throw SRParameterException("The src_name parameter cannot "\
                                   "be an empty string.");
    }

    if (dest_name.size() == 0) {
        throw SRParameterException("The dest_name parameter cannot "\
                                   "be an empty string.");
    }

    if (src_name == dest_name) {
        return;
    }

    copy_list(src_name, dest_name);
    delete_list(src_name);
}

// Get the length of the list
int Client::get_list_length(const std::string& list_name)
{
    // Build the list key
    std::string list_key = _build_list_key(list_name, false);

    // Build the command
    SingleKeyCommand cmd;
    cmd << "LLEN" << Keyfield(list_key);

    // Run the command
    CommandReply reply = _run(cmd);

    // Check for errors and return value
    if (reply.has_error() > 0)
        throw SRRuntimeException("LLEN command failed. The list "\
                                 "length could not be retrieved.");

    if (reply.redis_reply_type() != "REDIS_REPLY_INTEGER")
        throw SRRuntimeException("An unexpected type was returned for "
                                 "for list length.");

    int list_length = reply.integer();

    if (list_length < 0)
        throw SRRuntimeException("An invalid, negative value was "
                                 "returned for list length.");

    return list_length;
}

// Poll the list length (strictly equal)
bool Client::poll_list_length(const std::string& name, int list_length,
                              int poll_frequency_ms, int num_tries)
{
    // Enforce positive list length
    if (list_length < 0) {
        throw SRParameterException("A positive value for list_length "\
                                   "must be provided.");
    }

    return _poll_list_length(name, list_length, poll_frequency_ms,
                             num_tries, std::equal_to<int>());
}

// Poll the list length (strictly equal)
bool Client::poll_list_length_gte(const std::string& name, int list_length,
                                 int poll_frequency_ms, int num_tries)
{
    // Enforce positive list length
    if (list_length < 0) {
        throw SRParameterException("A positive value for list_length "\
                                   "must be provided.");
    }

    return _poll_list_length(name, list_length, poll_frequency_ms,
                             num_tries, std::greater_equal<int>());
}

// Poll the list length (strictly equal)
bool Client::poll_list_length_lte(const std::string& name, int list_length,
                                 int poll_frequency_ms, int num_tries)
{
    // Enforce positive list length
    if (list_length < 0) {
        throw SRParameterException("A positive value for list_length "\
                                   "must be provided.");
    }

    return _poll_list_length(name, list_length, poll_frequency_ms,
                             num_tries, std::less_equal<int>());

    return false;
}

// Retrieve datasets in aggregation list
std::vector<DataSet> Client::get_datasets_from_list(const std::string& list_name)
{
    if (list_name.size() == 0) {
        throw SRParameterException("The list name must have length "\
                                   "greater than zero");
    }

    return _get_dataset_list_range(list_name, 0, -1);
}

// Retrieve a subset of datsets in the aggregation list
std::vector<DataSet> Client::get_dataset_list_range(const std::string& list_name,
                                                    const int start_index,
                                                    const int end_index)
{
    if (list_name.size() == 0) {
        throw SRParameterException("The list name must have length "\
                                   "greater than zero");
    }

    return _get_dataset_list_range(list_name, start_index, end_index);
}

// Set the prefixes that are used for set and get methods using SSKEYIN
// and SSKEYOUT environment variables.
void Client::_set_prefixes_from_env()
{
    // Establish set prefix
    const char* keyout_p = std::getenv("SSKEYOUT");
    if (keyout_p != NULL)
        _put_key_prefix = keyout_p;
    else
        _put_key_prefix.clear();

    // Establish get prefix(es)
    char* keyin_p = std::getenv("SSKEYIN");
    if (keyin_p != NULL) {
        char* a = keyin_p;
        char* b = a;
        char parse_char = ',';
        while (*b != '\0') {
            if (*b == parse_char) {
                if (a != b)
                    _get_key_prefixes.push_back(std::string(a, b - a));
                a = ++b;
            }
            else
                b++;
        }
        if (a != b)
            _get_key_prefixes.push_back(std::string(a, b - a));
    }

    // Set the first prefix as the data source
    if (_get_key_prefixes.size() > 0)
        set_data_source(_get_key_prefixes[0].c_str());
}

// Get the key prefix for placement methods
inline std::string Client::_put_prefix()
{
    std::string prefix;
    if (_put_key_prefix.size() > 0)
        prefix =  _put_key_prefix + '.';
    return prefix;
}

// Get the key prefix for retrieval methods
inline std::string Client::_get_prefix()
{
    std::string prefix;
    if (_get_key_prefix.size() > 0)
        prefix =  _get_key_prefix + '.';
    return prefix;
}

// Append a vector of keys with the retrieval prefix
inline void Client::_append_with_get_prefix(std::vector<std::string>& keys)
{
    std::vector<std::string>::iterator prefix_it = keys.begin();
    for ( ; prefix_it != keys.end(); prefix_it++) {
        *prefix_it = _build_tensor_key(*prefix_it, true);
    }
}

// Append a vector of keys with the placement prefix
inline void Client::_append_with_put_prefix(std::vector<std::string>& keys)
{
    std::vector<std::string>::iterator prefix_it = keys.begin();
    for ( ; prefix_it != keys.end(); prefix_it++) {
        *prefix_it = _build_tensor_key(*prefix_it, false);
    }
}

// Execute the command to retrieve the DataSet metadata portion of the DataSet.
inline CommandReply Client::_get_dataset_metadata(const std::string& name)
{
    SingleKeyCommand cmd;
    cmd << "HGETALL" << Keyfield(_build_dataset_meta_key(name, true));
    return _run(cmd);
}

// Retrieve a tensor and add it to the dataset
inline void Client::_get_and_add_dataset_tensor(DataSet& dataset,
                                                const std::string& name,
                                                const std::string& key)
{
    // Run tensor retrieval command
    CommandReply reply = _redis_server->get_tensor(key);

    // Extract tensor properties from command reply
    std::vector<size_t> reply_dims = GetTensorCommand::get_dims(reply);
    std::string_view blob = GetTensorCommand::get_data_blob(reply);
    SRTensorType type = GetTensorCommand::get_data_type(reply);

    // Add tensor to the dataset
    dataset._add_to_tensorpack(name, (void*)blob.data(), reply_dims,
                               type, SRMemLayoutContiguous);
}

inline std::vector<DataSet>
Client::_get_dataset_list_range(const std::string& list_name,
                                const int start_index,
                                const int end_index)
{
    // Build the list key
    std::string list_key = _build_list_key(list_name, true);

    // Build the command to retrieve the list
    SingleKeyCommand cmd;
    cmd << "LRANGE" << Keyfield(list_key);
    cmd << std::to_string(start_index);
    cmd << std::to_string(end_index);

    // Run the command to retrive the list
    CommandReply reply = _run(cmd);

    // Check for reply errors and correct type
    if (reply.has_error() > 0)
        throw SRRuntimeException("GET command failed. The aggregation "\
                                 "list could not be retrieved.");

    if (reply.redis_reply_type() != "REDIS_REPLY_ARRAY")
        throw SRRuntimeException("An unexpected type was returned for "
                                 "for the aggregation list.");

    // Create CommandList for retrieving all metadata values in pipeline
    CommandList metadata_cmd_list;

    for (size_t i = 0; i < reply.n_elements(); i++) {
        // Check that the ith entry is a string (i.e. key)
        if (reply[i].redis_reply_type() != "REDIS_REPLY_STRING") {
            throw SRRuntimeException("Element " + std::to_string(i) +
                                     " in the aggregation list has an "\
                                     "unexpected type: " +
                                     reply.redis_reply_type());
        }

        // Check that the string length is not zero
        if(reply[i].str_len() == 0) {
            throw SRRuntimeException("Element " + std::to_string(i) +
                                     " contains an empty key, which is "\
                                     "not permitted.");
        }

        // Get the dataset key from the list entry
        std::string dataset_key_prefix(reply[i].str(), reply[i].str_len());

        // Build the metadata retrieval command
        SingleKeyCommand* metadata_cmd =
            metadata_cmd_list.add_command<SingleKeyCommand>();
        (*metadata_cmd) << "HGETALL" << Keyfield(dataset_key_prefix + ".meta");
    }

    // Run the commands via unordered pipeline
    PipelineReply metadata_replies =
        _redis_server->run_via_unordered_pipelines(metadata_cmd_list);


    // Start a lists of datasets that will be returned to the users
    std::vector<DataSet> dataset_list;

    // Command list for all tensorget commands
    CommandList tensor_cmd_list;

    for (size_t i = 0; i < metadata_replies.size(); i++) {

        // Shallow copy of the underlying PipelineReply entry
        CommandReply metadata_reply = metadata_replies[i];

        // Check if metadata_reply has any errors
        if (metadata_reply.has_error() > 0) {
            throw SRRuntimeException("An error was encountered in "\
                                        "metdata retrieval.");
        }

        std::string dataset_key =
            std::string(reply[i].str(), reply[i].str_len());
        std::string dataset_name =
            _get_dataset_name_from_list_entry(dataset_key);

        // Unpack the dataset to get tensor names
        dataset_list.push_back(DataSet(dataset_name));
        DataSet& dataset = dataset_list.back();

        // Unpack the metadata
        _unpack_dataset_metadata(dataset, metadata_reply);

        // Loop through tensor names in the dataset
        std::vector<std::string> tensor_names =
            dataset.get_tensor_names();

        for(size_t j = 0; j < tensor_names.size(); j++) {

            // Make the tensor key
            std::string tensor_key = dataset_key + "." +
                                     tensor_names[j];

            // Build the tensor retrieval cmd
            SingleKeyCommand* tensor_cmd =
                tensor_cmd_list.add_command<SingleKeyCommand>();

            (*tensor_cmd) << "AI.TENSORGET" << Keyfield(tensor_key)
                          << "META" << "BLOB";
        }
    }

    // Run the tensor get pipeline
    PipelineReply tensor_replies =
        _redis_server->run_via_unordered_pipelines(tensor_cmd_list);

    // Unpack tensor replies
    size_t tensor_reply_index = 0;
    for (size_t i = 0; i < dataset_list.size(); i++) {

        DataSet& dataset = dataset_list[i];

        std::vector<std::string> tensor_names =
            dataset_list[i].get_tensor_names();

        // Add the tensor replies as tensors
        for (size_t j = 0; j < tensor_names.size(); j++) {

            // Shallow copy of the pipeline reply for this tensor
            CommandReply tensor_reply = tensor_replies[tensor_reply_index];

            // Extract tensor properties from command reply
            std::vector<size_t> reply_dims = GetTensorCommand::get_dims(tensor_reply);
            std::string_view blob = GetTensorCommand::get_data_blob(tensor_reply);
            SRTensorType type = GetTensorCommand::get_data_type(tensor_reply);

            // Add tensor to the dataset (deep copy)
            dataset._add_to_tensorpack(tensor_names[j], (void*)blob.data(), reply_dims,
                                       type, SRMemLayoutContiguous);

            // Increment tensor reply index
            tensor_reply_index++;
        }
    }

    return dataset_list;
}

// Build full formatted key of a tensor, based on current prefix settings.
inline std::string Client::_build_tensor_key(const std::string& key,
                                             const bool on_db)
{
    std::string prefix;
    if (_use_tensor_prefix)
        prefix = on_db ? _get_prefix() : _put_prefix();

    return prefix + key;
}

// Build full formatted key of a model or a script,
// based on current prefix settings.
inline std::string Client::_build_model_key(const std::string& key,
                                            const bool on_db)
{
    std::string prefix;
    if (_use_model_prefix)
        prefix = on_db ? _get_prefix() : _put_prefix();

    return prefix + key;
}

// Build full formatted key of a dataset, based on current prefix settings.
inline std::string Client::_build_dataset_key(const std::string& dataset_name,
                                              const bool on_db)
{
    std::string prefix;
    if (_use_tensor_prefix)
        prefix = on_db ? _get_prefix() : _put_prefix();

    return prefix + "{" + dataset_name + "}";
}

// Create the key for putting or getting a DataSet tensor in the database
inline std::string
Client::_build_dataset_tensor_key(const std::string& dataset_name,
                                  const std::string& tensor_name,
                                  const bool on_db)
{
    return _build_dataset_key(dataset_name, on_db) + "." + tensor_name;
}

// Create the keys for putting or getting a DataSet tensors in the database
inline std::vector<std::string>
Client::_build_dataset_tensor_keys(const std::string& dataset_name,
                                   const std::vector<std::string>& tensor_names,
                                   const bool on_db)
{
    std::vector<std::string> dataset_tensor_keys;
    for (size_t i = 0; i < tensor_names.size(); i++) {
        dataset_tensor_keys.push_back(
            _build_dataset_tensor_key(dataset_name, tensor_names[i], on_db));
    }

    return dataset_tensor_keys;
}

// Create the key for putting or getting DataSet metadata in the database
inline std::string
Client::_build_dataset_meta_key(const std::string& dataset_name,
                                const bool on_db)
{
    return _build_dataset_key(dataset_name, on_db) + ".meta";
}

// Create the key for putting or getting aggregation list in the dataset
inline std::string
Client::_build_list_key(const std::string& list_name,
                                    const bool on_db)
{
    std::string prefix;
    if (_use_list_prefix)
        prefix = on_db ? _get_prefix() : _put_prefix();

    return prefix + list_name;
}


// Create the key to place an indicator in the database that the
// dataset has been successfully stored.
inline std::string
Client::_build_dataset_ack_key(const std::string& dataset_name,
                               const bool on_db)
{
    return _build_dataset_meta_key(dataset_name, on_db);
}

// Append the Command associated with placing DataSet metadata in
// the database to a CommandList
void Client::_append_dataset_metadata_commands(CommandList& cmd_list,
                                               DataSet& dataset)
{
    std::string meta_key = _build_dataset_meta_key(dataset.get_name(), false);

    std::vector<std::pair<std::string, std::string>> mdf =
        dataset.get_metadata_serialization_map();
    if (mdf.size() == 0) {
        throw SRRuntimeException("An attempt was made to put "\
                                 "a DataSet into the database that "\
                                 "does not contain any fields or "\
                                 "tensors.");
    }

    SingleKeyCommand* del_cmd = cmd_list.add_command<SingleKeyCommand>();
    *del_cmd << "DEL" << Keyfield(meta_key);

    SingleKeyCommand* cmd = cmd_list.add_command<SingleKeyCommand>();
    if (cmd == NULL) {
        throw SRRuntimeException("Failed to create SingleKeyCommand.");
    }
    *cmd << "HMSET" << Keyfield(meta_key);
    for (size_t i = 0; i < mdf.size(); i++) {
        *cmd << mdf[i].first << mdf[i].second;
    }
}

// Append the Command associated with placing DataSet tensors in the
// database to a CommandList
void Client::_append_dataset_tensor_commands(CommandList& cmd_list,
                                             DataSet& dataset)
{
    DataSet::tensor_iterator it = dataset.tensor_begin();
    for ( ; it != dataset.tensor_end(); it++) {
        TensorBase* tensor = *it;
        std::string tensor_key = _build_dataset_tensor_key(
            dataset.get_name(), tensor->name(), false);
        SingleKeyCommand* cmd = cmd_list.add_command<SingleKeyCommand>();
        *cmd << "AI.TENSORSET" << Keyfield(tensor_key) << tensor->type_str()
             << tensor->dims() << "BLOB" << tensor->buf();
    }
}

// Append the Command associated with acknowledging that the DataSet is complete
// (all put commands processed) to the CommandList
void Client::_append_dataset_ack_command(CommandList& cmd_list, DataSet& dataset)
{
    std::string key = _build_dataset_ack_key(dataset.get_name(), false);
    SingleKeyCommand* cmd = cmd_list.add_command<SingleKeyCommand>();
    *cmd << "HSET" << Keyfield(key) << _DATASET_ACK_FIELD << "1";
}

// Put the metadata fields embedded in a CommandReply into the DataSet
void Client::_unpack_dataset_metadata(DataSet& dataset, CommandReply& reply)
{
    // Make sure we have paired elements
    if ((reply.n_elements() % 2) != 0)
        throw SRRuntimeException("The DataSet metadata reply "\
                                 "contains the wrong number of "\
                                 "elements.");

    // Process each pair of response fields
    for (size_t i = 0; i < reply.n_elements(); i += 2) {
        std::string field_name(reply[i].str(), reply[i].str_len());
        if (field_name != _DATASET_ACK_FIELD) {
            dataset._add_serialized_field(field_name,
                                          reply[i + 1].str(),
                                          reply[i + 1].str_len());
        }
    }
}

// Retrieve the tensor from the DataSet and return a TensorBase object that
// can be used to return tensor information to the user. The returned
// TensorBase object has been dynamically allocated, but not yet tracked
// for memory management in any object.
TensorBase* Client::_get_tensorbase_obj(const std::string& name)
{
    // Fetch the tensor
    std::string get_key = _build_tensor_key(name, true);
    CommandReply reply = _redis_server->get_tensor(get_key);
    if (reply.has_error())
        throw SRRuntimeException("tensor retrieval failed");

    std::vector<size_t> dims = GetTensorCommand::get_dims(reply);
    if (dims.size() <= 0)
        throw SRRuntimeException("The number of dimensions of the "\
                                 "fetched tensor are invalid: " +
                                 std::to_string(dims.size()));

    SRTensorType type = GetTensorCommand::get_data_type(reply);
    std::string_view blob = GetTensorCommand::get_data_blob(reply);

    for (size_t i = 0; i < dims.size(); i++) {
        if (dims[i] <= 0) {
            throw SRRuntimeException("Dimension " +
                                     std::to_string(i) +
                                     "of the fetched tensor is not valid: " +
                                     std::to_string(dims[i]));
        }
    }

    TensorBase* ptr = NULL;
    try {
        switch (type) {
            case SRTensorTypeDouble:
                ptr = new Tensor<double>(get_key, (void*)blob.data(),
                                        dims, type, SRMemLayoutContiguous);
                break;
            case SRTensorTypeFloat:
                ptr = new Tensor<float>(get_key, (void*)blob.data(),
                                        dims, type, SRMemLayoutContiguous);
                break;
            case SRTensorTypeInt64:
                ptr = new Tensor<int64_t>(get_key, (void*)blob.data(),
                                        dims, type, SRMemLayoutContiguous);
                break;
            case SRTensorTypeInt32:
                ptr = new Tensor<int32_t>(get_key, (void*)blob.data(),
                                        dims, type, SRMemLayoutContiguous);
                break;
            case SRTensorTypeInt16:
                ptr = new Tensor<int16_t>(get_key, (void*)blob.data(),
                                        dims, type, SRMemLayoutContiguous);
                break;
            case SRTensorTypeInt8:
                ptr = new Tensor<int8_t>(get_key, (void*)blob.data(),
                                        dims, type, SRMemLayoutContiguous);
                break;
            case SRTensorTypeUint16:
                ptr = new Tensor<uint16_t>(get_key, (void*)blob.data(),
                                        dims, type, SRMemLayoutContiguous);
                break;
            case SRTensorTypeUint8:
                ptr = new Tensor<uint8_t>(get_key, (void*)blob.data(),
                                        dims, type, SRMemLayoutContiguous);
                break;
            default :
                throw SRInternalException("The database provided an invalid "\
                                          "TensorType to Client::_get_tensorbase_obj(). "\
                                          "The tensor could not be retrieved.");
                break;
        }
    }
    catch (std::bad_alloc& e) {
        throw SRBadAllocException("tensor");
    }
    return ptr;
}

// Determine datset name from aggregation list entry
std::string Client::_get_dataset_name_from_list_entry(std::string& dataset_key)
{
    size_t open_brace_pos = dataset_key.find_first_of('{');

    if (open_brace_pos == std::string::npos) {
        throw SRInternalException("An aggregation list entry could not be "\
                                  "converted to a DataSet name because "\
                                  "the { character is missing.");
    }

    size_t close_brace_pos = dataset_key.find_last_of('}');

    if (close_brace_pos == std::string::npos) {
        throw SRInternalException("An aggregation list entry could not be "\
                                  "converted to a DataSet name because "\
                                  "the } character is missing.");
    }

    if (open_brace_pos == close_brace_pos) {
        throw SRInternalException("An empty DataSet name was encountered.  "\
                                  "All aggregation list entries must be "\
                                  "non-empty");
    }

    open_brace_pos += 1;
    close_brace_pos -= 1;
    return dataset_key.substr(open_brace_pos,
                              close_brace_pos - open_brace_pos + 1);
}

// Poll aggregation list given a comparison function
bool Client::_poll_list_length(const std::string& name, int list_length,
                               int poll_frequency_ms, int num_tries,
                               std::function<bool(int,int)> comp_func)
{
    // Enforce positive list length
    if (list_length < 0) {
        throw SRParameterException("A positive value for list_length "\
                                   "must be provided.");
    }

    // Check for the requested list length, return if found
    for (int i = 0; i < num_tries; i++) {
        if (comp_func(get_list_length(name),list_length)) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_frequency_ms));
    }

    return false;
}