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

// Establish a dataset
void Client::put_dataset(DataSet& dataset)
{
    CommandList cmds;
    _append_dataset_metadata_commands(cmds, dataset);
    _append_dataset_tensor_commands(cmds, dataset);
    _append_dataset_ack_command(cmds, dataset);
    _run(cmds);
}

// Retrieve the current dataset
DataSet Client::get_dataset(const std::string& name)
{
    // Get the metadata message and construct DataSet
    CommandReply reply = _get_dataset_metadata(name);
    if (reply.n_elements() == 0) {
        throw SRRuntimeException("The requested DataSet " +
                                 name + " does not exist.");
    }

    DataSet dataset(name);
    _unpack_dataset_metadata(dataset, reply);

    std::vector<std::string> tensor_names = dataset.get_tensor_names();

    for(size_t i = 0; i < tensor_names.size(); i++) {
        std::string tensor_key =
            _build_dataset_tensor_key(name, tensor_names[i], true);
        CommandReply reply = this->_redis_server->get_tensor(tensor_key);
        std::vector<size_t> reply_dims = GetTensorCommand::get_dims(reply);
        std::string_view blob = GetTensorCommand::get_data_blob(reply);
        SRTensorType type = GetTensorCommand::get_data_type(reply);
        dataset._add_to_tensorpack(tensor_names[i],
                                   (void*)blob.data(), reply_dims,
                                   type, SRMemLayoutContiguous);
    }

    return dataset;
}

// Rename the current dataset
void Client::rename_dataset(const std::string& name,
                            const std::string& new_name)
{
    copy_dataset(name, new_name);
    delete_dataset(name);
}

// Clone the dataset to a new name
void Client::copy_dataset(const std::string& src_name,
                          const std::string& dest_name)
{
    // Extract metadata
    CommandReply reply = _get_dataset_metadata(src_name);
    if (reply.n_elements() == 0) {
        throw SRRuntimeException("The requested DataSet " +
                                 src_name + " does not exist.");
    }
    DataSet dataset(src_name);
    _unpack_dataset_metadata(dataset, reply);

    // Clone tensor keys
    std::vector<std::string> tensor_names = dataset.get_tensor_names();
    std::vector<std::string> tensor_src_names;
    std::vector<std::string> tensor_dest_names;
    for (size_t i = 0; i < tensor_names.size(); i++) {
        std::string key = _build_dataset_tensor_key(src_name, tensor_names[i],true);
        tensor_src_names.push_back(key);
        key = _build_dataset_tensor_key(dest_name, tensor_names[i],false);
        tensor_dest_names.push_back(key);
    }

    // Clone tensors
    _redis_server->copy_tensors(tensor_src_names, tensor_dest_names);

    // Update the DataSet name to the destination name
    // so we can reuse the object for placing metadata
    // and ack commands
    dataset.name = dest_name;
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

    // Build the delete command
    CompoundCommand cmd;
    cmd.add_field("DEL");
    cmd.add_field(_build_dataset_meta_key(dataset.name, true), true);

    // Add in all the tensors to be deleted
    std::vector<std::string> tensor_names = dataset.get_tensor_names();
    for (size_t i = 0; i < tensor_names.size(); i++) {
        std::string tensor_key(_build_dataset_tensor_key(dataset.name,
                                            tensor_names[i], true));
        cmd.add_field(tensor_key, true);
    }

    // Run the command
    reply = _run(cmd);
    if (reply.has_error())
        return; // Bail on error

    // Acknowledge the response
    CompoundCommand cmd_ack_key;
    std::string dataset_ack_key(_build_dataset_ack_key(name, false));
    cmd_ack_key.add_field("DEL");
    cmd_ack_key.add_field(dataset_ack_key, true);

    reply = _run(cmd_ack_key);
}

// Put a tensor into the database
void Client::put_tensor(const std::string& key,
                        void* data,
                        const std::vector<size_t>& dims,
                        const SRTensorType type,
                        const SRMemoryLayout mem_layout)
{
    std::string p_key = _build_tensor_key(key, false);

    TensorBase* tensor = NULL;
    try {
        switch (type) {
            case SRTensorTypeDouble:
                tensor = new Tensor<double>(p_key, data, dims, type, mem_layout);
                break;
            case SRTensorTypeFloat:
                tensor = new Tensor<float>(p_key, data, dims, type, mem_layout);
                break;
            case SRTensorTypeInt64:
                tensor = new Tensor<int64_t>(p_key, data, dims, type, mem_layout);
                break;
            case SRTensorTypeInt32:
                tensor = new Tensor<int32_t>(p_key, data, dims, type, mem_layout);
                break;
            case SRTensorTypeInt16:
                tensor = new Tensor<int16_t>(p_key, data, dims, type, mem_layout);
                break;
            case SRTensorTypeInt8:
                tensor = new Tensor<int8_t>(p_key, data, dims, type, mem_layout);
                break;
            case SRTensorTypeUint16:
                tensor = new Tensor<uint16_t>(p_key, data, dims, type, mem_layout);
                break;
            case SRTensorTypeUint8:
                tensor = new Tensor<uint8_t>(p_key, data, dims, type, mem_layout);
                break;
            default:
                throw SRRuntimeException("Invalid type for put_tensor");
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

// Get the tensor data, dimensions, and type for the provided tensor key.
// This function will allocate and retain management of the memory for the
// tensor data.
void Client::get_tensor(const std::string& key,
                        void*& data,
                        std::vector<size_t>& dims,
                        SRTensorType& type,
                        const SRMemoryLayout mem_layout)
{
    // Retrieve the TensorBase from the database
    TensorBase* ptr = _get_tensorbase_obj(key);

    // Set the user values
    dims = ptr->dims();
    type = ptr->type();
    data = ptr->data_view(mem_layout);

    // Hold the Tensor in memory for memory management
    _tensor_memory.add_tensor(ptr);
}

// Get the tensor data, dimensions, and type for the provided tensor key.
// This function will allocate and retain management of the memory for the
// tensor data and dimensions. This is a c-style interface for the tensor
// dimensions. Another function exists for std::vector dimensions.
 void Client::get_tensor(const std::string& key,
                        void*& data,
                        size_t*& dims,
                        size_t& n_dims,
                        SRTensorType& type,
                        const SRMemoryLayout mem_layout)
{

    std::vector<size_t> dims_vec;
    get_tensor(key, data, dims_vec, type, mem_layout);

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
void Client::unpack_tensor(const std::string& key,
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

    std::string get_key = _build_tensor_key(key, true);
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
                throw SRRuntimeException("Invalid type for unpack_tensor");
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

// Move a tensor from one key to another key
void Client::rename_tensor(const std::string& key,
                           const std::string& new_key)
{
    std::string p_key = _build_tensor_key(key, true);
    std::string p_new_key = _build_tensor_key(new_key, false);
    CommandReply reply = _redis_server->rename_tensor(p_key, p_new_key);
    if (reply.has_error())
        throw SRRuntimeException("rename_tensor failed");
}

// Delete a tensor from the database
void Client::delete_tensor(const std::string& key)
{
    std::string p_key = _build_tensor_key(key, true);
    CommandReply reply = _redis_server->delete_tensor(p_key);
    if (reply.has_error())
        throw SRRuntimeException("delete_tensor failed");
}

// Copy the tensor from the source key to the destination key
void Client::copy_tensor(const std::string& src_key,
                         const std::string& dest_key)
{
    std::string p_src_key = _build_tensor_key(src_key, true);
    std::string p_dest_key = _build_tensor_key(dest_key, false);
    CommandReply reply = _redis_server->copy_tensor(p_src_key, p_dest_key);
    if (reply.has_error())
        throw SRRuntimeException("copy_tensor failed");
}

// Set a model from file in the database for future execution
void Client::set_model_from_file(const std::string& key,
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
        throw SRRuntimeException("model_file is a required "
                                 "parameter of set_model.");
    }

    std::ifstream fin(model_file, std::ios::binary);
    std::ostringstream ostream;
    ostream << fin.rdbuf();

    const std::string tmp = ostream.str();
    std::string_view model(tmp.data(), tmp.length());

    set_model(key, model, backend, device, batch_size,
              min_batch_size, tag, inputs, outputs);
}

// Set a model from a string buffer in the database for future execution
void Client::set_model(const std::string& key,
                       const std::string_view& model,
                       const std::string& backend,
                       const std::string& device,
                       int batch_size,
                       int min_batch_size,
                       const std::string& tag,
                       const std::vector<std::string>& inputs,
                       const std::vector<std::string>& outputs)
{
    if (key.size() == 0) {
        throw SRRuntimeException("key is a required parameter of set_model.");
    }

    if (backend.size() == 0) {
        throw SRParameterException("backend is a required  "\
                                   "parameter of set_model.");
    }

    if (backend.compare("TF") != 0) {
        if (inputs.size() > 0) {
            throw SRRuntimeException("INPUTS in the model set command "\
                                     "is only valid for TF models");
        }
        if (outputs.size() > 0) {
            throw SRRuntimeException("OUTPUTS in the model set command "\
                                     "is only valid for TF models");
        }
    }

    const char* backends[] = { "TF", "TFLITE", "TORCH", "ONNX" };
    bool found = false;
    for (size_t i = 0; i < sizeof(backends)/sizeof(backends[0]); i++)
        found = found || (backend.compare(backends[i]) != 0);
    if (!found) {
        throw SRRuntimeException(backend + " is not a valid backend.");
    }

    if (device.size() == 0) {
        throw SRParameterException("device is a required "
                                   "parameter of set_model.");
    }

    if (device.compare("CPU") != 0 &&
        std::string(device).find("GPU") == std::string::npos) {
        throw SRRuntimeException(backend + " is not a valid backend.");
    }

    std::string p_key = _build_model_key(key, false);
    _redis_server->set_model(p_key, model, backend, device,
                             batch_size, min_batch_size,
                             tag, inputs, outputs);
}

// Retrieve the model from the database
std::string_view Client::get_model(const std::string& key)
{
    std::string get_key = _build_model_key(key, true);
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
void Client::set_script_from_file(const std::string& key,
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
    set_script(key, device, script);
}

// Set a script from a string buffer in the database for future execution
void Client::set_script(const std::string& key,
                        const std::string& device,
                        const std::string_view& script)
{
    std::string s_key = _build_model_key(key, false);
    _redis_server->set_script(s_key, device, script);
}

// Retrieve the script from the database
std::string_view Client::get_script(const std::string& key)
{
    std::string get_key = _build_model_key(key, true);
    CommandReply reply = _redis_server->get_script(get_key);
    char* script = _model_queries.allocate(reply.str_len());
    if (script == NULL)
        throw SRBadAllocException("model query");
    std::memcpy(script, reply.str(), reply.str_len());
    return std::string_view(script, reply.str_len());
}

// Run a model in the database using the specificed input and output tensors
void Client::run_model(const std::string& key,
                       std::vector<std::string> inputs,
                       std::vector<std::string> outputs)
{
    std::string get_key = _build_model_key(key, true);

    if (_use_tensor_prefix) {
        _append_with_get_prefix(inputs);
        _append_with_put_prefix(outputs);
    }
    _redis_server->run_model(get_key, inputs, outputs);
}

// Run a script function in the database using the specificed input and output tensors
void Client::run_script(const std::string& key,
                        const std::string& function,
                        std::vector<std::string> inputs,
                        std::vector<std::string> outputs)
{
    std::string get_key = _build_model_key(key, true);

    if (_use_tensor_prefix) {
        _append_with_get_prefix(inputs);
        _append_with_put_prefix(outputs);
    }
    _redis_server->run_script(get_key, function, inputs, outputs);
}

// Check if the key exists in the database
bool Client::key_exists(const std::string& key)
{
    return _redis_server->key_exists(key);
}

// Check if the tensor (or the dataset) exists in the database
bool Client::tensor_exists(const std::string& name)
{
    std::string get_key = _build_tensor_key(name, true);
    return _redis_server->key_exists(get_key);
}

// Check if the datyaset exists in the database
bool Client::dataset_exists(const std::string& name)
{
    // Same implementation as for tensors; the next line is NOT a type
    std::string g_key = _build_dataset_ack_key(name, true);
    return this->_redis_server->key_exists(g_key);
}

// Check if the model (or the script) exists in the database
bool Client::model_exists(const std::string& name)
{
    std::string get_key = _build_model_key(name, true);
    return _redis_server->model_key_exists(get_key);
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
        if (_get_key_prefixes[i].compare(source_id )== 0) {
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

    cmd.add_field("INFO");
    cmd.add_field("EVERYTHING");
    CommandReply reply = _run(cmd);
    if (reply.has_error())
        throw SRRuntimeException("INFO EVERYTHING command failed on server");

    // Parse the results
    return DBInfoCommand::parse_db_node_info(std::string(reply.str(),
                                                        reply.str_len()));
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

    cmd.add_field("CLUSTER");
    cmd.add_field("INFO");
    CommandReply reply = _run(cmd);
    if (reply.has_error())
        throw SRRuntimeException("CLUSTER INFO command failed on server");

    // Parse the results
    return ClusterInfoCommand::parse_db_cluster_info(std::string(reply.str(),
                                                     reply.str_len()));
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

    cmd.add_field("FLUSHDB");

    CommandReply reply = _run(cmd);
    if (reply.has_error() > 0)
        throw SRRuntimeException("FLUSHDB command failed");
}

// Read the configuration parameters of a running server
std::unordered_map<std::string,std::string> Client::config_get(std::string expression,
                                                               std::string address)
{
    AddressAtCommand cmd;
    std::string host = cmd.parse_host(address);
    uint64_t port = cmd.parse_port(address);

    cmd.set_exec_address_port(host, port);

    cmd.add_field("CONFIG");
    cmd.add_field("GET");
    cmd.add_field(expression);

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

    cmd.add_field("CONFIG");
    cmd.add_field("SET");
    cmd.add_field(config_param);
    cmd.add_field(value);

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
    cmd.add_field("SAVE");

    CommandReply reply = _run(cmd);
    if (reply.has_error() > 0)
        throw SRRuntimeException("SAVE command failed");
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
    cmd.add_field("HGETALL");
    cmd.add_field(_build_dataset_meta_key(name, true), true);
    return _run(cmd);
}

// Build full formatted key of a tensor, based on current prefix settings.
inline std::string Client::_build_tensor_key(const std::string& key, bool on_db)
{
    std::string prefix;
    if (_use_tensor_prefix)
        prefix = on_db ? _get_prefix() : _put_prefix();

    return prefix + key;
}

// Build full formatted key of a model or a script, based on current prefix settings.
inline std::string Client::_build_model_key(const std::string& key, bool on_db)
{
    std::string prefix;
    if (_use_model_prefix)
        prefix = on_db ? _get_prefix() : _put_prefix();

    return prefix + key;
}

// Build full formatted key of a dataset, based on current prefix settings.
inline std::string Client::_build_dataset_key(const std::string& dataset_name, bool on_db)
{
    std::string prefix;
    if (_use_tensor_prefix)
        prefix = on_db ? _get_prefix() : _put_prefix();

    return prefix + "{" + dataset_name + "}";
}

// Create the key for putting or getting a DataSet tensor in the database
inline std::string Client::_build_dataset_tensor_key(const std::string& dataset_name,
                                                     const std::string& tensor_name,
                                                     bool on_db)
{
    return _build_dataset_key(dataset_name, on_db) + "." + tensor_name;
}

// Create the key for putting or getting DataSet metadata in the database
inline std::string Client::_build_dataset_meta_key(const std::string& dataset_name,
                                                   bool on_db)
{
    return _build_dataset_key(dataset_name, on_db) + ".meta";
}

// Create the key to place an indicator in the database that the dataset has been
// successfully stored.
inline std::string Client::_build_dataset_ack_key(const std::string& dataset_name,
                                                  bool on_db)
{
    return _build_tensor_key(dataset_name, on_db);
}

// Append the Command associated with placing DataSet metadata in the database
// to a CommandList
void Client::_append_dataset_metadata_commands(CommandList& cmd_list,
                                               DataSet& dataset)
{
    std::string meta_key = _build_dataset_meta_key(dataset.name, false);

    std::vector<std::pair<std::string, std::string>> mdf =
        dataset.get_metadata_serialization_map();
    if (mdf.size() == 0) {
        throw SRRuntimeException("An attempt was made to put "\
                                 "a DataSet into the database that "\
                                 "does not contain any fields or "\
                                 "tensors.");
    }

    SingleKeyCommand* del_cmd = cmd_list.add_command<SingleKeyCommand>();
    del_cmd->add_field("DEL");
    del_cmd->add_field(meta_key, true);

    SingleKeyCommand* cmd = cmd_list.add_command<SingleKeyCommand>();
    if (cmd == NULL) {
        throw SRRuntimeException("Failed to create singlekeycommande");
    }
    cmd->add_field("HMSET");
    cmd->add_field (meta_key, true);
    for (size_t i = 0; i < mdf.size(); i++) {
        cmd->add_field(mdf[i].first);
        cmd->add_field(mdf[i].second);
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
            dataset.name, tensor->name(), false);
        SingleKeyCommand* cmd = cmd_list.add_command<SingleKeyCommand>();
        cmd->add_field("AI.TENSORSET");
        cmd->add_field(tensor_key, true);
        cmd->add_field(tensor->type_str());
        cmd->add_fields(tensor->dims());
        cmd->add_field("BLOB");
        cmd->add_field_ptr(tensor->buf());
    }
}

// Append the Command associated with acknowledging that the DataSet is complete
// (all put commands processed) to the CommandList
void Client::_append_dataset_ack_command(CommandList& cmd_list, DataSet& dataset)
{
    std::string dataset_ack_key = _build_dataset_ack_key(dataset.name, false);

    // SingleKeyCommand* cmd = new SingleKeyCommand();
    SingleKeyCommand* cmd = cmd_list.add_command<SingleKeyCommand>();
    cmd->add_field("SET");
    cmd->add_field(dataset_ack_key, true);
    cmd->add_field("1");
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
        dataset._add_serialized_field(field_name,
                                      reply[i + 1].str(),
                                      reply[i + 1].str_len());
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
                                     "of the fetched tensor is "\
                                     "not valid: " +
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
                throw SRRuntimeException("An invalid TensorType was "\
                                         "provided to "
                                         "Client::_get_tensorbase_obj(). "
                                         "The tensor could not be "\
                                         "retrieved.");
                break;
        }
    }
    catch (std::bad_alloc& e) {
        throw SRBadAllocException("tensor");
    }
    return ptr;
}
