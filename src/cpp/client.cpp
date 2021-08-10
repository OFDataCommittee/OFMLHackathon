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

using namespace SmartRedis;

Client::Client(bool cluster)
{
    if(cluster) {
        this->_redis_cluster = new RedisCluster();
        this->_redis = 0;
        this->_redis_server = this->_redis_cluster;
    }
    else {
        this->_redis_cluster = 0;
        this->_redis = new Redis();
        this->_redis_server = this->_redis;
    }
    this->_set_prefixes_from_env();

    this->_use_tensor_prefix = true;
    this->_use_model_prefix = false;

    return;
}

Client::~Client() {
    if(this->_redis_cluster)
        delete this->_redis_cluster;
    if(this->_redis)
        delete this->_redis;
}

void Client::put_dataset(DataSet& dataset)
{
    CommandList cmds;
    this->_append_dataset_metadata_commands(cmds, dataset);
    this->_append_dataset_tensor_commands(cmds, dataset);
    this->_append_dataset_ack_command(cmds, dataset);
    this->_run(cmds);
    return;
}

DataSet Client::get_dataset(const std::string& name)
{
    // Get the metadata message and construct DataSet
    CommandReply reply = this->_get_dataset_metadata(name);

    if(reply.n_elements()==0)
        throw std::runtime_error("The requested DataSet " +
                                 name + " does not exist.");

    DataSet dataset = DataSet(name);
    this->_unpack_dataset_metadata(dataset, reply);

    std::vector<std::string> tensor_names = dataset.get_tensor_names();

    std::vector<size_t> reply_dims;
    std::string_view blob;
    TensorType type;
    std::string tensor_key;
    for(size_t i=0; i<tensor_names.size(); i++) {
        tensor_key =
            this->_build_dataset_tensor_key(name, tensor_names[i],
                                            true);
        Command cmd;
        cmd.add_field("AI.TENSORGET");
        cmd.add_field(tensor_key, true);
        cmd.add_field("META");
        cmd.add_field("BLOB");
        reply = this->_run(cmd);
        reply_dims = CommandReplyParser::get_tensor_dims(reply);
        blob = CommandReplyParser::get_tensor_data_blob(reply);
        type = CommandReplyParser::get_tensor_data_type(reply);
        dataset._add_to_tensorpack(tensor_names[i],
                                   (void*)blob.data(), reply_dims,
                                   type, MemoryLayout::contiguous);
    }
    return dataset;
}

void Client::rename_dataset(const std::string& name,
                            const std::string& new_name)
{
    this->copy_dataset(name, new_name);
    this->delete_dataset(name);
    return;
}

void Client::copy_dataset(const std::string& src_name,
                          const std::string& dest_name)
{
    CommandReply reply = this->_get_dataset_metadata(src_name);
    DataSet dataset = DataSet(src_name);
    this->_unpack_dataset_metadata(dataset, reply);

    std::vector<std::string> tensor_names = dataset.get_tensor_names();
    std::vector<std::string> tensor_src_names;
    std::vector<std::string> tensor_dest_names;
    for(size_t i=0; i<tensor_names.size(); i++) {
        tensor_src_names.push_back(
            this->_build_dataset_tensor_key(src_name,
                                            tensor_names[i],true));
        tensor_dest_names.push_back(
            this->_build_dataset_tensor_key(dest_name,
                                            tensor_names[i],false));
    }
    this->_redis_server->copy_tensors(tensor_src_names,
                                      tensor_dest_names);

    // Update the DataSet name to the destination name
    // so we can reuse the object for placing metadata
    // and ack commands
    dataset.name = dest_name;
    CommandList put_meta_cmds;
    CommandReply put_meta_reply;
    this->_append_dataset_metadata_commands(put_meta_cmds, dataset);
    this->_append_dataset_ack_command(put_meta_cmds, dataset);
    put_meta_reply = this->_run(put_meta_cmds);

    return;
}

void Client::delete_dataset(const std::string& name)
{
    CommandReply reply = this->_get_dataset_metadata(name);

    DataSet dataset = DataSet(name);
    std::string field_name;
    this->_unpack_dataset_metadata(dataset, reply);

    Command cmd;
    cmd.add_field("DEL");
    cmd.add_field(this->_build_dataset_meta_key(dataset.name, true),true);

    std::vector<std::string> tensor_names = dataset.get_tensor_names();
    std::string tensor_key;
    for(size_t i=0; i<tensor_names.size(); i++) {
        tensor_key =
            this->_build_dataset_tensor_key(dataset.name,
                                            tensor_names[i], true);
        cmd.add_field(tensor_key, true);
    }

    reply = this->_run(cmd);

    Command cmd_ack_key;
    std::string dataset_ack_key =
        this->_build_dataset_ack_key(name, false);
    cmd_ack_key.add_field("DEL");
    cmd_ack_key.add_field(dataset_ack_key, true);

    reply = this->_run(cmd_ack_key);

    return;
}

void Client::put_tensor(const std::string& key,
                        void* data,
                        const std::vector<size_t>& dims,
                        const TensorType type,
                        const MemoryLayout mem_layout)
{
    std::string p_key = this->_build_tensor_key(key, false);

    TensorBase* tensor;
    switch(type) {
        case TensorType::dbl :
            tensor = new Tensor<double>(p_key, data, dims,
                                        type, mem_layout);
            break;
        case TensorType::flt :
            tensor = new Tensor<float>(p_key, data, dims,
                                        type, mem_layout);
            break;
        case TensorType::int64 :
            tensor = new Tensor<int64_t>(p_key, data, dims,
                                         type, mem_layout);
            break;
        case TensorType::int32 :
            tensor = new Tensor<int32_t>(p_key, data, dims,
                                         type, mem_layout);
            break;
        case TensorType::int16 :
            tensor = new Tensor<int16_t>(p_key, data, dims,
                                         type, mem_layout);
            break;
        case TensorType::int8 :
            tensor = new Tensor<int8_t>(p_key, data, dims,
                                        type, mem_layout);
            break;
        case TensorType::uint16 :
            tensor = new Tensor<uint16_t>(p_key, data, dims,
                                          type, mem_layout);
            break;
        case TensorType::uint8 :
            tensor = new Tensor<uint8_t>(p_key, data, dims,
                                         type, mem_layout);
            break;
        default : // FINDME: Need to handle this case better
            throw std::runtime_error("Invalid type for put_tensor");
    }

    CommandReply reply = this->_redis_server->put_tensor(*tensor);

    delete tensor;
}

void Client::get_tensor(const std::string& key,
                        void*& data,
                        std::vector<size_t>& dims,
                        TensorType& type,
                        const MemoryLayout mem_layout)
{
    // Retrieve the TensorBase from the database
    TensorBase* ptr = this->_get_tensorbase_obj(key);

    // Set the user values
    dims = ptr->dims();
    type = ptr->type();
    data = ptr->data_view(mem_layout);

    // Hold the Tensor in memory for memory management
    this->_tensor_memory.add_tensor(ptr);

    return;
}

void Client::get_tensor(const std::string& key,
                                void*& data,
                                size_t*& dims,
                                size_t& n_dims,
                                TensorType& type,
                                const MemoryLayout mem_layout)
{

    std::vector<size_t> dims_vec;
    this->get_tensor(key, data, dims_vec,
                    type, mem_layout);

    size_t dims_bytes = sizeof(size_t)*dims_vec.size();
    dims = this->_dim_queries.allocate_bytes(dims_bytes);
    n_dims = dims_vec.size();

    std::vector<size_t>::const_iterator it = dims_vec.cbegin();
    std::vector<size_t>::const_iterator it_end = dims_vec.cend();
    size_t i = 0;
    while(it!=it_end) {
        dims[i] = *it;
        i++;
        it++;
    }
    return;
}

void Client::unpack_tensor(const std::string& key,
                          void* data,
                          const std::vector<size_t>& dims,
                          const TensorType type,
                          const MemoryLayout mem_layout)
{
    if(mem_layout == MemoryLayout::contiguous &&
        dims.size()>1) {
        throw std::runtime_error("The destination memory space "\
                                "dimension vector should only "\
                                "be of size one if the memory "\
                                "layout is contiguous.");
        }

    std::string g_key = this->_build_tensor_key(key, true);
    CommandReply reply = this->_redis_server->get_tensor(g_key);

    std::vector<size_t> reply_dims =
        CommandReplyParser::get_tensor_dims(reply);

    if(mem_layout == MemoryLayout::contiguous ||
        mem_layout == MemoryLayout::fortran_contiguous) {

        int total_dims = 1;
        for(size_t i=0; i<reply_dims.size(); i++) {
            total_dims *= reply_dims[i];
        }
        if( (total_dims != dims[0]) &&
            (mem_layout == MemoryLayout::contiguous) )
        throw std::runtime_error("The dimensions of the fetched "\
                                "tensor do not match the length of "\
                                "the contiguous memory space.");
    }

    if(mem_layout == MemoryLayout::nested) {
        if(dims.size()!= reply_dims.size())
        throw std::runtime_error("The number of dimensions of the  "\
                                "fetched tensor, " +
                                std::to_string(reply_dims.size()) + " "\
                                "does not match the number of "\
                                "dimensions of the user memory space, " +
                                std::to_string(dims.size()));

        for(size_t i=0; i<reply_dims.size(); i++) {
            if(dims[i]!=reply_dims[i]) {
                throw std::runtime_error("The dimensions of the fetched tensor "\
                                        "do not match the provided "\
                                        "dimensions of the user memory space.");
            }
        }
    }

    TensorType reply_type = CommandReplyParser::get_tensor_data_type(reply);
    if(type!=reply_type)
        throw std::runtime_error("The type of the fetched tensor "\
                                "does not match the provided type");
    std::string_view blob = CommandReplyParser::get_tensor_data_blob(reply);

    TensorBase* tensor;
    switch(reply_type) {
        case TensorType::dbl :
        tensor = new Tensor<double>(g_key, (void*)blob.data(),
                                    reply_dims, reply_type,
                                    MemoryLayout::contiguous);
        break;
        case TensorType::flt :
        tensor = new Tensor<float>(g_key, (void*)blob.data(),
                                    reply_dims, reply_type,
                                    MemoryLayout::contiguous);
        break;
        case TensorType::int64  :
        tensor = new Tensor<int64_t>(g_key, (void*)blob.data(),
                                    reply_dims, reply_type,
                                    MemoryLayout::contiguous);
        break;
        case TensorType::int32 :
        tensor = new Tensor<int32_t>(g_key, (void*)blob.data(),
                                    reply_dims, reply_type,
                                    MemoryLayout::contiguous);
        break;
        case TensorType::int16 :
        tensor = new Tensor<int16_t>(g_key, (void*)blob.data(),
                                    reply_dims, reply_type,
                                    MemoryLayout::contiguous);
        break;
        case TensorType::int8 :
        tensor = new Tensor<int8_t>(g_key, (void*)blob.data(),
                                    reply_dims, reply_type,
                                    MemoryLayout::contiguous);
        break;
        case TensorType::uint16 :
        tensor = new Tensor<uint16_t>(g_key, (void*)blob.data(),
                                      reply_dims, reply_type,
                                      MemoryLayout::contiguous);
        break;
        case TensorType::uint8 :
        tensor = new Tensor<uint8_t>(g_key, (void*)blob.data(),
                                     reply_dims, reply_type,
                                     MemoryLayout::contiguous);
        break;
        default : // FINDME: Need to handle this case better
            throw std::runtime_error("Invalid type for unpack_tensor");
    }

    tensor->fill_mem_space(data, dims, mem_layout);
    delete tensor;
    return;
}

void Client::rename_tensor(const std::string& key,
                           const std::string& new_key)
{
    std::string p_key = this->_build_tensor_key(key, true);
    std::string p_new_key = this->_build_tensor_key(new_key, false);
    CommandReply reply =
        this->_redis_server->rename_tensor(p_key, p_new_key);
    return;
}

void Client::delete_tensor(const std::string& key)
{
    std::string p_key = this->_build_tensor_key(key, true);
    CommandReply reply =
            this->_redis_server->delete_tensor(p_key);
    return;
}

void Client::copy_tensor(const std::string& src_key,
                         const std::string& dest_key)
{
    std::string p_src_key = this->_build_tensor_key(src_key, true);
    std::string p_dest_key = this->_build_tensor_key(dest_key, false);
    CommandReply reply =
            this->_redis_server->copy_tensor(p_src_key, p_dest_key);
    return;
}

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
    if(model_file.size()==0)
        throw std::runtime_error("model_file is a required  "
                                "parameter of set_model.");

    std::ifstream fin(model_file, std::ios::binary);
    std::ostringstream ostream;
    ostream << fin.rdbuf();

    const std::string tmp = ostream.str();
    std::string_view model(tmp.data(), tmp.length());

    this->set_model(key, model, backend, device, batch_size,
                    min_batch_size, tag, inputs, outputs);
    return;
}

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
    if(key.size()==0)
        throw std::runtime_error("key is a required parameter "
                                 "of set_model.");

    if(backend.size()==0)
        throw std::runtime_error("backend is a required  "
                                 "parameter of set_model.");

    if(backend.compare("TF")!=0) {
        if(inputs.size() > 0)
        throw std::runtime_error("INPUTS in the model set command "\
                                 "is only valid for TF models");
        if(outputs.size() > 0)
        throw std::runtime_error("OUTPUTS in the model set command "\
                                 "is only valid for TF models");
    }

    if(backend.compare("TF")!=0 && backend.compare("TFLITE")!=0 &&
        backend.compare("TORCH")!=0 && backend.compare("ONNX")!=0) {
        throw std::runtime_error(std::string(backend) +
                                    " is not a valid backend.");
    }

    if(device.size()==0)
        throw std::runtime_error("device is a required  "
                                 "parameter of set_model.");

    if(device.compare("CPU")!=0 &&
        std::string(device).find("GPU")==std::string::npos) {
        throw std::runtime_error(std::string(backend) +
                                 " is not a valid backend.");
    }

    std::string p_key = this->_build_model_key(key, false);
    this->_redis_server->set_model(p_key, model, backend, device,
                                   batch_size, min_batch_size,
                                   tag, inputs, outputs);

    return;
}

std::string_view Client::get_model(const std::string& key)
{
    std::string g_key = this->_build_model_key(key, true);
    CommandReply reply = this->_redis_server->get_model(g_key);
    char* model = this->_model_queries.allocate(reply.str_len());
    std::memcpy(model, reply.str(), reply.str_len());
    return std::string_view(model, reply.str_len());
}

void Client::set_script_from_file(const std::string& key,
                                  const std::string& device,
                                  const std::string& script_file)
{
    std::ifstream fin(script_file);
    std::ostringstream ostream;
    ostream << fin.rdbuf();

    const std::string tmp = ostream.str();
    std::string_view script(tmp.data(), tmp.length());

    this->set_script(key, device, script);
    return;
}


void Client::set_script(const std::string& key,
                        const std::string& device,
                        const std::string_view& script)
{
    std::string s_key = this->_build_model_key(key, false);
    this->_redis_server->set_script(s_key, device, script);
    return;
}

std::string_view Client::get_script(const std::string& key)
{
    std::string g_key = this->_build_model_key(key, true);
    CommandReply reply = this->_redis_server->get_script(g_key);
    char* script = this->_model_queries.allocate(reply.str_len());
    std::memcpy(script, reply.str(), reply.str_len());
    return std::string_view(script, reply.str_len());
}

void Client::run_model(const std::string& key,
                       std::vector<std::string> inputs,
                       std::vector<std::string> outputs)
{
    std::string g_key = this->_build_model_key(key, true);

    if (this->_use_tensor_prefix) {
        this->_append_with_get_prefix(inputs);
        this->_append_with_put_prefix(outputs);
    }
    this->_redis_server->run_model(g_key, inputs, outputs);
    return;
}

void Client::run_script(const std::string& key,
                        const std::string& function,
                        std::vector<std::string> inputs,
                        std::vector<std::string> outputs)
{
    std::string g_key = this->_build_model_key(key, true);

    if (this->_use_tensor_prefix) {
        this->_append_with_get_prefix(inputs);
        this->_append_with_put_prefix(outputs);
    }
    this->_redis_server->run_script(g_key, function, inputs, outputs);
    return;
}

bool Client::key_exists(const std::string& key)
{
    return this->_redis_server->key_exists(key);
}

bool Client::tensor_exists(const std::string& name)
{
    std::string g_key = this->_build_tensor_key(name, true);
    return this->_redis_server->key_exists(g_key);
}

bool Client::model_exists(const std::string& name)
{
    std::string g_key = this->_build_model_key(name, true);
    return this->_redis_server->model_key_exists(g_key);
}

bool Client::poll_key(const std::string& key,
                      int poll_frequency_ms,
                      int num_tries)
{
    bool key_exists = false;

    while(!(num_tries==0)) {
        if(this->key_exists(key)) {
            key_exists = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_frequency_ms));
        if(num_tries>0)
        num_tries--;
    }

    return key_exists;
}

bool Client::poll_model(const std::string& name,
                        int poll_frequency_ms,
                        int num_tries)
{
    bool key_exists = false;

    while(!(num_tries==0)) {
        if(this->model_exists(name)) {
            key_exists = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_frequency_ms));
        if(num_tries>0)
        num_tries--;
    }

    return key_exists;
}

bool Client::poll_tensor(const std::string& name,
                         int poll_frequency_ms,
                         int num_tries)
{
    bool key_exists = false;

    while(!(num_tries==0)) {
        if(this->tensor_exists(name)) {
            key_exists = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_frequency_ms));
        if(num_tries>0)
        num_tries--;
    }

    if(key_exists)
        return true;
    else
        return false;
}

void Client::set_data_source(std::string source_id)
{
    bool valid_prefix = false;
    int num_prefix = _get_key_prefixes.size();
    int i = 0;
    for (i=0; i<num_prefix; i++) {
        if (this->_get_key_prefixes[i].compare(source_id)==0) {
            valid_prefix = true;
            break;
        }
    }

    if (valid_prefix)
        this->_get_key_prefix = this->_get_key_prefixes[i];
    else
        throw std::runtime_error("Client error: data source " +
                    std::string(source_id) +
                    "could not be found during client "+
                    "initialization.");
}


void Client::use_model_ensemble_prefix(bool use_prefix)
{
    this->_use_model_prefix = use_prefix;
}

void Client::use_tensor_ensemble_prefix(bool use_prefix)
{
    this->_use_tensor_prefix = use_prefix;
}

parsed_reply_nested_map Client::get_db_node_info(std::string address)
{
    std::string host = address.substr(0, address.find(":"));
    uint64_t port = std::stoul (address.substr(address.find(":") + 1),
                                nullptr, 0);
    if (host.empty() or port == 0)
        throw std::runtime_error(std::string(address) +
                                 "is not a valid database node address.");
    Command cmd;
    cmd.set_exec_address_port(host, port);
    cmd.add_field("INFO");
    cmd.add_field("everything");
    CommandReply reply = this->_run(cmd);
    return CommandReplyParser::parse_db_node_info(std::string(reply.str(),
                                                        reply.str_len()));
}

parsed_reply_map Client::get_db_cluster_info(std::string address)
{
    if(this->_redis_cluster == NULL)
        return parsed_reply_map();

    std::string host = address.substr(0, address.find(":"));
    uint64_t port = std::stoul (address.substr(address.find(":") + 1),
                                nullptr, 0);
    if (host.empty() or port == 0)
        throw std::runtime_error(std::string(address) +
                                 "is not a valid database node address.");
    Command cmd;
    cmd.set_exec_address_port(host, port);
    cmd.add_field("CLUSTER");
    cmd.add_field("INFO");
    CommandReply reply = this->_run(cmd);
    return CommandReplyParser::parse_db_cluster_info(std::string(reply.str(),
                                                     reply.str_len()));
}

CommandReply Client::_run(Command& cmd)
{
    return this->_redis_server->run(cmd);
}

CommandReply Client::_run(CommandList& cmds)
{
    return this->_redis_server->run(cmds);
}

void Client::_set_prefixes_from_env()
{
    const char* keyout_p = std::getenv("SSKEYOUT");
    if (keyout_p)
        this->_put_key_prefix = keyout_p;
    else
        this->_put_key_prefix.clear();

    char* keyin_p = std::getenv("SSKEYIN");
    if(keyin_p) {
        char* a = &keyin_p[0];
        char* b = a;
        char parse_char = ',';
        while (*b) {
            if(*b==parse_char) {
                if(a!=b)
                    this->_get_key_prefixes.push_back(std::string(a, b-a));
                a=++b;
            }
            else
                b++;
        }
        if(a!=b)
            this->_get_key_prefixes.push_back(std::string(a, b-a));
    }

    if (this->_get_key_prefixes.size() > 0)
        this->set_data_source(this->_get_key_prefixes[0].c_str());
}

inline std::string Client::_put_prefix()
{
    std::string prefix;
    if(this->_put_key_prefix.size()>0)
        prefix =  this->_put_key_prefix + '.';
    return prefix;
}

inline std::string Client::_get_prefix()
{
    std::string prefix;
    if(this->_get_key_prefix.size()>0)
        prefix =  this->_get_key_prefix + '.';
    return prefix;
}

inline void Client::_append_with_get_prefix(
                     std::vector<std::string>& keys)
{
    std::vector<std::string>::iterator prefix_it;
    std::vector<std::string>::iterator prefix_it_end;
    prefix_it = keys.begin();
    prefix_it_end = keys.end();
    while(prefix_it != prefix_it_end) {
        *prefix_it = this->_build_tensor_key(*prefix_it, true);
        prefix_it++;
    }
    return;
}

inline void Client::_append_with_put_prefix(
                     std::vector<std::string>& keys)
{
    std::vector<std::string>::iterator prefix_it;
    std::vector<std::string>::iterator prefix_it_end;
    prefix_it = keys.begin();
    prefix_it_end = keys.end();
    while(prefix_it != prefix_it_end) {
        *prefix_it = this->_build_tensor_key(*prefix_it, false);
        prefix_it++;
    }
    return;
}

inline CommandReply Client::_get_dataset_metadata(const std::string& name)
{
    Command cmd;
    cmd.add_field("HGETALL");
    cmd.add_field(this->_build_dataset_meta_key(name, true), true);
    return this->_run(cmd);
}

inline std::string Client::_build_tensor_key(const std::string& key, bool on_db)
{
    std::string prefix;
    if (this->_use_tensor_prefix)
        prefix = on_db ? this->_get_prefix() : this->_put_prefix();

    return prefix + key;
}

inline std::string Client::_build_model_key(const std::string& key, bool on_db)
{
    std::string prefix;
    if (this->_use_model_prefix)
        prefix = on_db ? this->_get_prefix() : this->_put_prefix();

    return prefix + key;
}

inline std::string Client::_build_dataset_key(const std::string& dataset_name, bool on_db)
{
    std::string key;
    std::string prefix;
    if (this->_use_tensor_prefix)
        prefix = on_db ? this->_get_prefix() : this->_put_prefix();

    key = prefix +
          "{" + dataset_name + "}";
    return key;
}

inline std::string Client::_build_dataset_tensor_key(const std::string& dataset_name,
                                                     const std::string& tensor_name,
                                                     bool on_db)
{
    return this->_build_dataset_key(dataset_name, on_db) + "."+ tensor_name;
}

inline std::string Client::_build_dataset_meta_key(const std::string& dataset_name,
                                                   bool on_db)
{
    return this->_build_dataset_key(dataset_name, on_db) + ".meta";
}

inline std::string Client::_build_dataset_ack_key(const std::string& dataset_name,
                                                  bool on_db)
{
    return this->_build_tensor_key(dataset_name, on_db);
}

void Client::_append_dataset_metadata_commands(CommandList& cmd_list,
                                               DataSet& dataset)
{
    std::string meta_key =
        this->_build_dataset_meta_key(dataset.name, false);

    std::vector<std::pair<std::string, std::string>>
        mdf = dataset.get_metadata_serialization_map();

    if(mdf.size()==0)
        throw std::runtime_error("An attempt was made to put "\
                                 "a DataSet into the database that "\
                                 "does not contain any fields or "\
                                 "tensors.");

    Command* cmd = cmd_list.add_command();
    cmd->add_field("HMSET");
    cmd->add_field (meta_key, true);
    for(size_t i=0; i<mdf.size(); i++) {
        cmd->add_field(mdf[i].first);
        cmd->add_field(mdf[i].second);
    }
    return;
}

void Client::_append_dataset_tensor_commands(CommandList& cmd_list,
                                             DataSet& dataset)
{
    DataSet::tensor_iterator it = dataset.tensor_begin();
    DataSet::tensor_iterator it_end = dataset.tensor_end();
    TensorBase* tensor = 0;
    std::string tensor_key;

    Command* cmd;
    while(it != it_end) {
        tensor = *it;
        tensor_key =
            this->_build_dataset_tensor_key(dataset.name,
                                            tensor->name(),
                                            false);
        cmd = cmd_list.add_command();
        cmd->add_field("AI.TENSORSET");
        cmd->add_field(tensor_key, true);
        cmd->add_field(tensor->type_str());
        cmd->add_fields(tensor->dims());
        cmd->add_field("BLOB");
        cmd->add_field_ptr(tensor->buf());
        it++;
    }
    return;
}

void Client::_append_dataset_ack_command(CommandList& cmd_list,
                                         DataSet& dataset)
{
    std::string dataset_ack_key =
        this->_build_dataset_ack_key(dataset.name, false);

    Command* cmd = cmd_list.add_command();
    cmd->add_field("SET");
    cmd->add_field(dataset_ack_key, true);
    cmd->add_field("1");
    return;
}

void Client::_unpack_dataset_metadata(DataSet& dataset,
                                      CommandReply& reply)
{
    std::string field_name;

    if(reply.n_elements()%2)
        throw std::runtime_error("The DataSet metadata reply "\
                                  "contains the wrong number of "\
                                  "elements.");

    for(size_t i=0; i<reply.n_elements(); i+=2) {
        field_name = std::string(reply[i].str(), reply[i].str_len());
        dataset._add_serialized_field(field_name,
                                      reply[i+1].str(),
                                      reply[i+1].str_len());
    }
    return;
}

TensorBase* Client::_get_tensorbase_obj(const std::string& name)
{
    std::string g_key = this->_build_tensor_key(name, true);

    CommandReply reply = this->_redis_server->get_tensor(g_key);

    std::vector<size_t> dims =
        CommandReplyParser::get_tensor_dims(reply);

    TensorType type =
         CommandReplyParser::get_tensor_data_type(reply);

    std::string_view blob =
        CommandReplyParser::get_tensor_data_blob(reply);

    if(dims.size()<=0)
        throw std::runtime_error("The number of dimensions of the "\
                                "fetched tensor are invalid: " +
                                std::to_string(dims.size()));

    for(size_t i=0; i<dims.size(); i++) {
        if(dims[i]<=0) {
        throw std::runtime_error("Dimension " +
                                 std::to_string(i) +
                                 "of the fetched tensor is "\
                                 "not valid: " +
                                 std::to_string(dims[i]));
        }
    }

    TensorBase* ptr;
    switch(type) {
        case TensorType::dbl :
            ptr = new Tensor<double>(g_key, (void*)blob.data(), dims,
                                     type, MemoryLayout::contiguous);

            break;
        case TensorType::flt :
            ptr = new Tensor<float>(g_key, (void*)blob.data(), dims,
                                    type, MemoryLayout::contiguous);
            break;
        case TensorType::int64 :
            ptr = new Tensor<int64_t>(g_key, (void*)blob.data(), dims,
                                     type, MemoryLayout::contiguous);
            break;
        case TensorType::int32 :
            ptr = new Tensor<int32_t>(g_key, (void*)blob.data(), dims,
                                      type, MemoryLayout::contiguous);
            break;
        case TensorType::int16 :
            ptr = new Tensor<int16_t>(g_key, (void*)blob.data(), dims,
                                      type, MemoryLayout::contiguous);
            break;
        case TensorType::int8 :
            ptr = new Tensor<int8_t>(g_key, (void*)blob.data(), dims,
                                     type, MemoryLayout::contiguous);
            break;
        case TensorType::uint16 :
            ptr = new Tensor<uint16_t>(g_key, (void*)blob.data(), dims,
                                       type, MemoryLayout::contiguous);
            break;
        case TensorType::uint8 :
            ptr = new Tensor<uint8_t>(g_key, (void*)blob.data(), dims,
                                      type, MemoryLayout::contiguous);
            break;
        default :
            throw std::runtime_error("An invalid TensorType was "\
                                     "provided to "
                                     "Client::_get_tensorbase_obj(). "
                                     "The tensor could not be "\
                                     "retrieved.");
            break;
    }
    return ptr;
}
