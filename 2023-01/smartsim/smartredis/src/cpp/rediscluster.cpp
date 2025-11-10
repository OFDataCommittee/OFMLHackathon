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

#include <sw/redis++/redis++.h>
#include "rediscluster.h"
#include "nonkeyedcommand.h"
#include "keyedcommand.h"
#include "srexception.h"
#include "utility.h"
#include "srobject.h"
#include "configoptions.h"

using namespace SmartRedis;

// RedisCluster constructor
RedisCluster::RedisCluster(ConfigOptions* cfgopts)
    : RedisServer(cfgopts)
{
    SRAddress db_address(_get_ssdb());
    if (!db_address._is_tcp) {
        throw SRRuntimeException("Unix Domain Socket is not supported with clustered Redis");
    }
    _is_domain_socket = false;
    _connect(db_address);
    _map_cluster();
    if (_address_node_map.count(db_address.to_string()) > 0)
        _last_prefix = _address_node_map.at(db_address.to_string())->prefix;
    else if (_db_nodes.size() > 0)
        _last_prefix = _db_nodes[0].prefix;
    else
        throw SRRuntimeException("Cluster mapping failed in client initialization");
}

// RedisCluster constructor. Uses address provided to constructor instead of
// environment variables
RedisCluster::RedisCluster(ConfigOptions* cfgopts, std::string address_spec)
    : RedisServer(cfgopts)
{
    SRAddress db_address(address_spec);
    _connect(db_address);
    _map_cluster();
    if (_address_node_map.count(db_address.to_string()) > 0)
        _last_prefix = _address_node_map.at(db_address.to_string())->prefix;
    else if (_db_nodes.size() > 0)
        _last_prefix = _db_nodes[0].prefix;
    else
        throw SRRuntimeException("Cluster mapping failed in client initialization");
}

// RedisCluster destructor
RedisCluster::~RedisCluster()
{
    if (_redis_cluster != NULL) {
        delete _redis_cluster;
        _redis_cluster = NULL;
    }
}

// Run a single-key Command on the server
CommandReply RedisCluster::run(SingleKeyCommand& cmd)
{
    // Preprend the target database to the command
    std::string db_prefix;
    if (cmd.has_keys())
        db_prefix = _get_db_node_prefix(cmd);
    else
        throw SRRuntimeException("Redis has failed to find database");

    return _run(cmd, db_prefix);
}

// Run a compound Command on the server
CommandReply RedisCluster::run(CompoundCommand& cmd)
{
    std::string db_prefix;
    if (cmd.has_keys())
        db_prefix = _get_db_node_prefix(cmd);
    else
        throw SRRuntimeException("Redis has failed to find database");

    return _run(cmd, db_prefix);
}

// Run a MultiKeyCommand on the server
CommandReply RedisCluster::run(MultiKeyCommand& cmd)
{
    std::string db_prefix;
    if (cmd.has_keys())
        db_prefix = _get_db_node_prefix(cmd);
    else
        throw SRRuntimeException("Redis has failed to find database");

    return _run(cmd, db_prefix);
}

// Run a non-keyed Command that addresses the given db node on the server
CommandReply RedisCluster::run(AddressAtCommand& cmd)
{
    std::string db_prefix;
    SRAddress address(cmd.get_address());
    if (is_addressable(address))
        db_prefix = _address_node_map.at(address.to_string())->prefix;
    else
        throw SRRuntimeException("Redis has failed to find database");

    return _run(cmd, db_prefix);
}

// Run a non-keyed Command that addresses any db node on the server
CommandReply RedisCluster::run(AddressAnyCommand &cmd)
{
    return _run(cmd, _last_prefix);
}

// Run a non-keyed Command that addresses any db node on the server
CommandReply RedisCluster::run(AddressAllCommand &cmd)
{
    // Bounds check the key_index
    if (cmd.key_index != -1 && cmd.get_field_count() < cmd.key_index) {
        throw SRInternalException("Invalid key_index executing command!");
    }

    // Find the segment to be tweaked for each node
    std::string field;
    if (cmd.key_index != -1) {
        Command::const_iterator it = cmd.cbegin();
        int i = 0;
        for ( ; it != cmd.cend(); it++) {
            if (i == cmd.key_index)
                field = std::string(it->data(), it->size());
            i++;
        }
    }

    // Loop through all nodes to execute the command at each
    std::vector<DBNode>::const_iterator node = _db_nodes.cbegin();
    CommandReply reply;
    for ( ; node != _db_nodes.cend(); node++) {
        // swap in a replacement segment for each one
        std::string new_field = "{" + node->prefix + "}." + field;
        cmd.set_field_at(new_field, cmd.key_index, true);

        // Execute the updated command
        cmd.set_exec_address(node->address);
        reply = _run(cmd, node->prefix);
        if (reply.has_error() > 0)
            break; // Short-circuit failure on error
    }
    return reply;
}

// Run multiple single-key or single-hash slot Command on the server.
// Each Command in the CommandList is run sequentially
std::vector<CommandReply> RedisCluster::run(CommandList& cmds)
{
    std::vector<CommandReply> replies;
    CommandList::iterator cmd = cmds.begin();
    for ( ; cmd != cmds.end(); cmd++) {
        replies.push_back(dynamic_cast<Command*>(*cmd)->run_me(this));
    }
    return replies;
}

// Run multiple single-key or single-hash slot Command on the server.
PipelineReply RedisCluster::run_via_unordered_pipelines(CommandList& cmd_list)
{
    // Map for shard index to Command indices so we can track order of execution
    std::vector<std::vector<size_t>> shard_cmd_index_list(_db_nodes.size());

    // Map for shard index to Command pointers so pipelines can be easily rebuilt
    std::vector<std::vector<Command*>> shard_cmds(_db_nodes.size());

    // Calculate shard for execution of each Command
    CommandList::iterator cmd = cmd_list.begin();
    size_t cmd_num = 0;

    for ( ; cmd != cmd_list.end(); cmd++, cmd_num++) {

        // Make sure we have at least one key
        if ((*cmd)->has_keys() == false) {
            throw SRInternalException("Only single key commands are supported "\
                                      "by RedisCluster::run_via_unordered"\
                                      "_pipelines.");
        }

        // Get keys for the command
        std::vector<std::string> keys = (*cmd)->get_keys();

        // Check that there is only one key
        if (keys.size() != 1) {
            throw SRInternalException("Only single key commands are supported "\
                                      "by RedisCluster::run_via_unordered_"\
                                      "pipelines.");
        }

        // Get the shard index for the first key
        size_t db_index = _get_db_node_index(keys[0]);

        // Push back the command index to the shard list of commands
        shard_cmd_index_list[db_index].push_back(cmd_num);

        // Push back a pointer to the command for pipeline construction
        shard_cmds[db_index].push_back(*cmd);
    }

    // Define an empty PipelineReply object to store all shard replies
    PipelineReply all_replies;

    // Keep track of CommandList index order of execution (ooe)
    std::vector<size_t> cmd_list_index_ooe;
    cmd_list_index_ooe.reserve(cmd_list.size());

    volatile size_t pipeline_completion_count = 0;
    size_t num_shards = shard_cmd_index_list.size();
    Exception error_response = Exception("no error");
    bool success_status[num_shards];
    std::mutex results_mutex;

    // Loop over all shards and execute pipelines
    for (size_t s = 0; s < num_shards; s++) {
        // Get shard prefix
        std::string shard_prefix = _db_nodes[s].prefix;

        // Only execute if there are commands
        if (shard_cmd_index_list[s].size() == 0) {
            success_status[s] = true;
            {
                // Avoid race condition on update of completion_count
                std::unique_lock<std::mutex> lock(results_mutex);
                ++pipeline_completion_count;
            } // End scope and release lock
            continue;
        }

        // Submit a task to execute these commands
        _tp->submit_job([this, &shard_cmds, s, shard_prefix, &success_status,
                         &results_mutex, &cmd_list_index_ooe,
                         &shard_cmd_index_list, &all_replies,
                         &pipeline_completion_count, &error_response]() mutable
        {
            // Run the pipeline, catching any exceptions thrown
            PipelineReply reply;
            try {
                reply = _run_pipeline(shard_cmds[s], shard_prefix);
                success_status[s] = true;
            }
            catch (Exception& e) {
                error_response = e;
                success_status[s] = false;
            }

            // Acquire the lock to store our results
            {
                std::unique_lock<std::mutex> results_lock(results_mutex);

                // Skip storing results if we hit an error
                if (success_status[s]) {
                    // Add the CommandList indices into vector for later reordering
                    cmd_list_index_ooe.insert(cmd_list_index_ooe.end(),
                                            shard_cmd_index_list[s].begin(),
                                            shard_cmd_index_list[s].end());

                    // Store results
                    all_replies += std::move(reply);
                }

                // Increment completion counter
                ++pipeline_completion_count;
            }  // Release the lock
        });
    }

    // Wait until all jobs have finished
    while (pipeline_completion_count != num_shards)
        ; // Spin

    // Throw an exception if one was generated in processing the threads
    for (size_t i = 0; i < num_shards; i++) {
        if (!success_status[i]) {
            throw error_response;
        }
    }

    // Reorder the command replies in all_replies to align
    // with order of execution
    all_replies.reorder(cmd_list_index_ooe);

    return all_replies;
}

// Check if a model or script key exists in the database
bool RedisCluster::model_key_exists(const std::string& key)
{
    // Add model prefix to the key
    DBNode* node = &(_db_nodes[0]);
    if (node == NULL)
        return false;
    std::string prefixed_key = '{' + node->prefix + "}." + key;

    // And perform key existence check
    return key_exists(prefixed_key);
}

// Check if a key exists in the database
bool RedisCluster::key_exists(const std::string& key)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd << "EXISTS" << Keyfield(key);

    // Run it
    CommandReply reply = run(cmd);
    if (reply.has_error() > 0)
        throw SRRuntimeException("Error encountered while checking "\
                                 "for existence of key " + key);
    return (bool)reply.integer();
}

// Check if a hash field exists in the database
bool RedisCluster::hash_field_exists(const std::string& key,
                                     const std::string& field)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd << "HEXISTS" << Keyfield(key) << field;

    // Run it
    CommandReply reply = run(cmd);
    if (reply.has_error() > 0)
        throw SRRuntimeException("Error encountered while checking "\
                                 "for existence of hash field " +
                                 field + " at key " + key);
    return (bool)reply.integer();
}

// Check if a key exists in the database
bool RedisCluster::is_addressable(const SRAddress& address) const
{
    return _address_node_map.find(address.to_string()) !=
        _address_node_map.end();
}

// Put a Tensor on the server
CommandReply RedisCluster::put_tensor(TensorBase& tensor)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd << "AI.TENSORSET" << Keyfield(tensor.name()) << tensor.type_str()
        << tensor.dims() << "BLOB" << tensor.buf();

    // Run it
    return run(cmd);
}

// Get a Tensor from the server
CommandReply RedisCluster::get_tensor(const std::string& key)
{
    // Build the command
    GetTensorCommand cmd;
    cmd << "AI.TENSORGET"<< Keyfield(key) << "META" << "BLOB";

    // Run it
    return run(cmd);
}

// Get a list of Tensor from the server
PipelineReply RedisCluster::get_tensors(const std::vector<std::string>& keys)
{
    // Build up the commands to get the tensors
    CommandList cmdlist; // This just holds the memory
    std::vector<Command*> cmds;
    for (auto it = keys.begin(); it != keys.end(); ++it) {
        GetTensorCommand* cmd = cmdlist.add_command<GetTensorCommand>();
        (*cmd) << "AI.TENSORGET" << Keyfield(*it) << "META" << "BLOB";
        cmds.push_back(cmd);
    }

    // Get the shard index for the first key
    size_t db_index = _get_db_node_index(keys[0]);
    std::string shard_prefix = _db_nodes[db_index].prefix;

    // Run them via pipeline
    return _run_pipeline(cmds, shard_prefix);
}

// Rename a tensor in the database
CommandReply RedisCluster::rename_tensor(const std::string& key,
                                         const std::string& new_key)
{
    // Check whether we have to switch hash slots
    uint16_t key_hash_slot = _get_hash_slot(key);
    uint16_t new_key_hash_slot = _get_hash_slot(new_key);

    // If not, we can use a simple RENAME command
    CommandReply reply;
    if (key_hash_slot == new_key_hash_slot) {
        // Build the command
        CompoundCommand cmd;
        cmd << "RENAME" << Keyfield(key) << Keyfield(new_key);

        // Run it
        reply = run(cmd);
    }

    // Otherwise we need to clone the tensor and then nuke the old one
    else {
        copy_tensor(key, new_key);
        reply = delete_tensor(key);
    }

    // Done
    return reply;
}

// Delete a tensor in the database
CommandReply RedisCluster::delete_tensor(const std::string& key)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd << "UNLINK" << Keyfield(key);

    // Run it
    return run(cmd);
}

// Copy a tensor from the source key to the destination key
CommandReply RedisCluster::copy_tensor(const std::string& src_key,
                                       const std::string& dest_key)
{
    //TODO can we do COPY for same hash slot or database (only for redis 6.2)?

    // Build the GET command
    GetTensorCommand cmd_get;
    cmd_get << "AI.TENSORGET" << Keyfield(src_key) << "META" << "BLOB";

    // Run the GET command
    CommandReply cmd_get_reply = run(cmd_get);
    if (cmd_get_reply.has_error() > 0)
        throw SRRuntimeException("Failed to find tensor " + src_key);

    // Decode the tensor
    std::vector<size_t> dims = cmd_get.get_dims(cmd_get_reply);
    std::string_view blob = cmd_get.get_data_blob(cmd_get_reply);
    SRTensorType type = cmd_get.get_data_type(cmd_get_reply);

    // Build the PUT command
    MultiKeyCommand cmd_put;
    cmd_put << "AI.TENSORSET" << Keyfield(dest_key) << TENSOR_STR_MAP.at(type)
            << dims << "BLOB" << blob;

    // Run the PUT command
    return run(cmd_put);
}

// Copy a vector of tensors from source keys to destination keys
CommandReply RedisCluster::copy_tensors(const std::vector<std::string>& src,
                                        const std::vector<std::string>& dest)
{
    // Make sure vectors are the same length
    if (src.size() != dest.size()) {
        throw SRRuntimeException("differing size vectors "\
                                 "passed to copy_tensors");
    }

    // Copy tensors one at a time. We only need to check one iterator
    // for reaching the end since we know from above that they are the
    // same length
    std::vector<std::string>::const_iterator it_src = src.cbegin();
    std::vector<std::string>::const_iterator it_dest = dest.cbegin();
    CommandReply reply;
    for ( ; it_src != src.cend(); it_src++, it_dest++) {
        reply = copy_tensor(*it_src, *it_dest);
        if (reply.has_error() > 0) {
            throw SRRuntimeException("tensor copy failed");
        }

    }

    // Done
    return reply;
}

// Set a model from a string buffer in the database for future execution
CommandReply RedisCluster::set_model(const std::string& model_name,
                                     const std::vector<std::string_view>& model,
                                     const std::string& backend,
                                     const std::string& device,
                                     int batch_size,
                                     int min_batch_size,
                                     int min_batch_timeout,
                                     const std::string& tag,
                                     const std::vector<std::string>& inputs,
                                     const std::vector<std::string>& outputs)
{
    // Build the basic command
    CommandReply reply;
    AddressAllCommand cmd;
    cmd.key_index = 1;
    cmd << "AI.MODELSTORE" << Keyfield(model_name) << backend << device;

    // Add optional fields as requested
    if (tag.size() > 0) {
        cmd << "TAG" << tag;
    }
    if (batch_size > 0) {
        cmd << "BATCHSIZE" << std::to_string(batch_size);
    }
    if (min_batch_size > 0) {
        cmd << "MINBATCHSIZE" << std::to_string(min_batch_size);
    }
    if (min_batch_timeout > 0) {
        cmd << "MINBATCHTIMEOUT" << std::to_string(min_batch_timeout);
    }
    if ( inputs.size() > 0) {
        cmd << "INPUTS" << std::to_string(inputs.size()) << inputs;
    }
    if (outputs.size() > 0) {
        cmd << "OUTPUTS" << std::to_string(outputs.size()) << outputs;
    }
    cmd << "BLOB" << model;

    // Run it
    reply = run(cmd);
    if (reply.has_error() > 0) {
        throw SRRuntimeException("set_model failed!");
    }

    // Done
    return reply;
}

// Set a model from std::string_view buffer in the
// database for future execution in a multi-GPU system
void RedisCluster::set_model_multigpu(const std::string& name,
                                      const std::vector<std::string_view>& model,
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
    // Store a copy of the model for each GPU
    for (int i = first_gpu; i < num_gpus; i++) {
        // Set up parameters for this copy of the script
        std::string device = "GPU:" + std::to_string(i);
        std::string model_key = name + "." + device;

        // Store it
        CommandReply result = set_model(
            model_key, model, backend, device, batch_size, min_batch_size,
            min_batch_timeout, tag, inputs, outputs);
        if (result.has_error() > 0) {
            throw SRRuntimeException("Failed to set model for " + device);
        }
    }

    // Add a version for get_model to find
    CommandReply result = set_model(
        name, model, backend, "GPU", batch_size, min_batch_size,
        min_batch_timeout, tag, inputs, outputs);
    if (result.has_error() > 0) {
        throw SRRuntimeException("Failed to set general model");
    }
}

// Set a script from a string buffer in the database for future execution
CommandReply RedisCluster::set_script(const std::string& key,
                                      const std::string& device,
                                      std::string_view script)
{
    // Build the basic command
    CommandReply reply;
    AddressAllCommand cmd;
    cmd.key_index = 1;
    cmd << "AI.SCRIPTSET" << Keyfield(key) << device << "SOURCE" << script;

    // Run it
    reply = run(cmd);
    if (reply.has_error() > 0) {
        throw SRRuntimeException("set_script failed!");
    }

    // Done
    return reply;
}

// Set a script from std::string_view buffer in the
// database for future execution in a multi-GPU system
void RedisCluster::set_script_multigpu(const std::string& name,
                                       const std::string_view& script,
                                       int first_gpu,
                                       int num_gpus)
{
    // Store a copy of the script for each GPU
    for (int i = first_gpu; i < num_gpus; i++) {
        // Set up parameters for this copy of the script
        std::string device = "GPU:" + std::to_string(i);
        std::string script_key = name + "." + device;

        // Store it
        CommandReply result = set_script(script_key, device, script);
        if (result.has_error() > 0) {
            throw SRRuntimeException("Failed to set script for " + device);
        }
    }

    // Add a copy for get_script to find
    CommandReply result = set_script(name, "GPU", script);
    if (result.has_error() > 0) {
        throw SRRuntimeException("Failed to set general script");
    }
}

// Run a model in the database using the specified input and output tensors
CommandReply RedisCluster::run_model(const std::string& model_name,
                                     std::vector<std::string> inputs,
                                     std::vector<std::string> outputs)
{
    // Check for a non-default timeout setting
    int run_timeout;
    get_config_integer(run_timeout, _MODEL_TIMEOUT_ENV_VAR,
                       _DEFAULT_MODEL_TIMEOUT);

    /*  For this version of run model, we have to copy all
        input and output tensors, so we will randomly select
        a model.  We can't use rand, because MPI would then
        have the same random number across all ranks.  Instead
        We will choose it based on the db of the first input tensor.
    */

    uint16_t hash_slot = _get_hash_slot(inputs[0]);
    uint16_t db_index = _db_node_hash_search(hash_slot, 0, _db_nodes.size()-1);
    DBNode* db = &(_db_nodes[db_index]);
    if (db == NULL) {
        throw SRRuntimeException("Missing DB node found in run_model");
    }

    // Generate temporary names so that all keys go to same slot
    std::vector<std::string> tmp_inputs = _get_tmp_names(inputs, db->prefix);
    std::vector<std::string> tmp_outputs = _get_tmp_names(outputs, db->prefix);

    // Copy all input tensors to temporary names to align hash slots
    copy_tensors(inputs, tmp_inputs);

    // Use the model on our selected node
    std::string model_key = "{" + db->prefix + "}." + std::string(model_name);

    // Build the MODELRUN command
    CompoundCommand cmd;
    cmd << "AI.MODELEXECUTE" << Keyfield(model_key)
        << "INPUTS" << std::to_string(tmp_inputs.size()) << tmp_inputs
        << "OUTPUTS" << std::to_string(tmp_outputs.size()) << tmp_outputs
        << "TIMEOUT" << std::to_string(run_timeout);

    // Run it
    CommandReply reply = run(cmd);
    if (reply.has_error() > 0) {
        std::string error("run_model failed for node ");
        error += db_index;
        throw SRRuntimeException(error);
    }

    // Store the outputs back to the database
    copy_tensors(tmp_outputs, outputs);

    // Clean up the temp keys
    std::vector<std::string> keys_to_delete;
    keys_to_delete.insert(
        keys_to_delete.end(), tmp_outputs.begin(), tmp_outputs.end());
    keys_to_delete.insert(
        keys_to_delete.end(), tmp_inputs.begin(), tmp_inputs.end());
    _delete_keys(keys_to_delete);

    // Done
    return reply;
}

// Run a model in the database using the
// specified input and output tensors in a multi-GPU system
void RedisCluster::run_model_multigpu(const std::string& name,
                                      std::vector<std::string> inputs,
                                      std::vector<std::string> outputs,
                                      int offset,
                                      int first_gpu,
                                      int num_gpus)
{
    int gpu = first_gpu + _modulo(offset, num_gpus);
    std::string device = "GPU:" + std::to_string(gpu);
    std::string target_model = name + "." + device;
    CommandReply result = run_model(target_model, inputs, outputs);
    if (result.has_error() > 0) {
        throw SRRuntimeException(
            "An error occurred while executing the model on " + device);
    }
}

// Run a script function in the database using the specified input
// and output tensors
CommandReply RedisCluster::run_script(const std::string& key,
                                      const std::string& function,
                                      std::vector<std::string> inputs,
                                      std::vector<std::string> outputs)
{
    // Locate the DB node for the script
    uint16_t hash_slot = _get_hash_slot(inputs[0]);
    uint16_t db_index = _db_node_hash_search(hash_slot, 0, _db_nodes.size() - 1);
    DBNode* db = &(_db_nodes[db_index]);
    if (db == NULL) {
        throw SRRuntimeException("Missing DB node found in run_script");
    }

    // Generate temporary names so that all keys go to same slot
    std::vector<std::string> tmp_inputs = _get_tmp_names(inputs, db->prefix);
    std::vector<std::string> tmp_outputs = _get_tmp_names(outputs, db->prefix);

    // Copy all input tensors to temporary names to align hash slots
    copy_tensors(inputs, tmp_inputs);
    std::string script_name = "{" + db->prefix + "}." + std::string(key);

    // Build the SCRIPTRUN command
    CompoundCommand cmd;
    cmd << "AI.SCRIPTRUN" << Keyfield(script_name) << function
        << "INPUTS" << tmp_inputs << "OUTPUTS" << tmp_outputs;

    // Run it
    CommandReply reply = run(cmd);
    if (reply.has_error() > 0) {
        std::string error("run_model failed for node ");
        error += db_index;
        throw SRRuntimeException(error);
    }

    // Store the output back to the database
    copy_tensors(tmp_outputs, outputs);

    // Clean up temp keys
    std::vector<std::string> keys_to_delete;
    keys_to_delete.insert(keys_to_delete.end(),
                            tmp_outputs.begin(),
                            tmp_outputs.end());
    keys_to_delete.insert(keys_to_delete.end(),
                            tmp_inputs.begin(),
                            tmp_inputs.end());
    _delete_keys(keys_to_delete);

    // Done
    return reply;
}

/*!
*   \brief Run a script function in the database using the
*          specified input and output tensors in a multi-GPU system
*   \param name The name associated with the script
*   \param function The name of the function in the script to run
*   \param inputs The names of input tensors to use in the script
*   \param outputs The names of output tensors that will be used
*                  to save script results
*   \param offset index of the current image, such as a processor
*                   ID or MPI rank
*   \param num_gpus the number of gpus for which the script was stored
*   \throw RuntimeException for all client errors
*/
void RedisCluster::run_script_multigpu(const std::string& name,
                                       const std::string& function,
                                       std::vector<std::string>& inputs,
                                       std::vector<std::string>& outputs,
                                       int offset,
                                       int first_gpu,
                                       int num_gpus)
{
    int gpu = first_gpu + _modulo(offset, num_gpus);
    std::string device = "GPU:" + std::to_string(gpu);
    std::string target_script = name + "." + device;
    CommandReply result = run_script(target_script, function, inputs, outputs);
    if (result.has_error() > 0) {
        throw SRRuntimeException(
            "An error occurred while executing the script on " + device);
    }
}

// Delete a model from the database
CommandReply RedisCluster::delete_model(const std::string& key)
{
    // Build the command
    CommandReply reply;
    AddressAllCommand cmd;
    cmd.key_index = 1;
    cmd << "AI.MODELDEL" << Keyfield(key);

    // Run it
    reply = run(cmd);
    if (reply.has_error() > 0) {
        throw SRRuntimeException("delete_model failed!");
    }

    // Done
    return reply;
}

// Remove a model from the database that was stored
// for use with multiple GPUs
void RedisCluster::delete_model_multigpu(
    const std::string& name, int first_gpu, int num_gpus)
{
    // Remove a copy of the model for each GPU
    CommandReply result;
    for (int i = first_gpu; i < num_gpus; i++) {
        std::string device = "GPU:" + std::to_string(i);
        std::string model_key = name + "." + device;
        result = delete_model(model_key);
        if (result.has_error() > 0) {
            throw SRRuntimeException("Failed to remove model for GPU " + std::to_string(i));
        }
    }

    // Remove the copy that was added for get_model to find
    result = delete_model(name);
    if (result.has_error() > 0) {
        throw SRRuntimeException("Failed to remove general model");
    }
}

// Delete a script from the database
CommandReply RedisCluster::delete_script(const std::string& key)
{
    CommandReply reply;
    AddressAllCommand cmd;
    cmd.key_index = 1;
    cmd << "AI.SCRIPTDEL" << Keyfield(key);

    // Run it
    reply = run(cmd);
    if (reply.has_error() > 0) {
        throw SRRuntimeException("delete_script failed!");
    }

    // Done
    return reply;
}

// Remove a script from the database that was stored
// for use with multiple GPUs
void RedisCluster::delete_script_multigpu(
    const std::string& name, int first_gpu, int num_gpus)
{
    // Remove a copy of the script for each GPU
    CommandReply result;
    for (int i = first_gpu; i < num_gpus; i++) {
        std::string device = "GPU:" + std::to_string(i);
        std::string script_key = name + "." + device;
        result = delete_script(script_key);
        if (result.has_error() > 0) {
            throw SRRuntimeException("Failed to remove script for GPU " + std::to_string(i));
        }
    }

    // Remove the copy that was added for get_script to find
    result = delete_script(name);
    if (result.has_error() > 0) {
        throw SRRuntimeException("Failed to remove general script");
    }
}

// Retrieve the model from the database
CommandReply RedisCluster::get_model(const std::string& key)
{
    // Build the node prefix
    std::string prefixed_str = "{" + _db_nodes[0].prefix + "}." + key;

    // Build the MODELGET command
    SingleKeyCommand cmd;
    cmd << "AI.MODELGET" << Keyfield(prefixed_str) << "BLOB";

    // Run it
    return run(cmd);
}

// Retrieve the script from the database
CommandReply RedisCluster::get_script(const std::string& key)
{
    // Build the node prefix
    std::string prefixed_str = "{" + _db_nodes[0].prefix + "}." + key;

    // Build the SCRIPTGET command
    SingleKeyCommand cmd;
    cmd << "AI.SCRIPTGET" << Keyfield(prefixed_str) << "SOURCE";

    // Run it
    return run(cmd);
}

// Retrieve the model and script AI.INFO
CommandReply RedisCluster::get_model_script_ai_info(const std::string& address,
                                                    const std::string& key,
                                                    const bool reset_stat)
{
    AddressAtCommand cmd;
    SRAddress db_address(address);

    // Determine the prefix we need for the model or script
    if (!is_addressable(db_address)) {
        throw SRRuntimeException("The provided address does "\
                                 "not match a cluster shard address.");
    }

    std::string db_prefix = _address_node_map.at(db_address.to_string())->prefix;
    std::string prefixed_key = "{" + db_prefix + "}." + key;

    // Build the Command
    cmd.set_exec_address(db_address);
    cmd << "AI.INFO" << Keyfield(prefixed_key);

    // Optionally add RESETSTAT to the command
    if (reset_stat) {
        cmd << "RESETSTAT";
    }

    return run(cmd);
}

// Retrieve the current model chunk size
int RedisCluster::get_model_chunk_size()
{
    // If we've already set a chunk size, just return it
    if (_model_chunk_size != _UNKNOWN_MODEL_CHUNK_SIZE)
        return _model_chunk_size;

    // Build the command
    AddressAnyCommand cmd;
    cmd << "AI.CONFIG" << "GET" << "MODEL_CHUNK_SIZE";

    CommandReply reply = run(cmd);
    if (reply.has_error() > 0)
        throw SRRuntimeException("AI.CONFIG GET MODEL_CHUNK_SIZE command failed");

    if (reply.redis_reply_type() != "REDIS_REPLY_INTEGER")
        throw SRRuntimeException("An unexpected type was returned for "
                                 "for the model chunk size.");

    int chunk_size = reply.integer();

    if (chunk_size < 0)
        throw SRRuntimeException("An invalid, negative value was "
                                 "returned for the model chunk size.");

    return chunk_size;
}

// Reconfigure the model chunk size for the database
void RedisCluster::set_model_chunk_size(int chunk_size)
{
    // Repeat for each server node:
    auto node = _db_nodes.cbegin();
    for ( ; node != _db_nodes.cend(); node++) {
        // Pick a node for the command
        AddressAtCommand cmd;
        cmd.set_exec_address(node->address);
        // Build the command
        cmd << "AI.CONFIG" << "MODEL_CHUNK_SIZE" << std::to_string(chunk_size);

        // Run it
        CommandReply reply = run(cmd);
        if (reply.has_error() > 0) {
            throw SRRuntimeException("set_model_chunk_size failed for node " + node->name);
        }
    }

    // Store the new model chunk size for later
    _model_chunk_size = chunk_size;
}

inline CommandReply RedisCluster::_run(const Command& cmd, std::string db_prefix)
{
    std::string_view sv_prefix(db_prefix.data(), db_prefix.size());

    // Execute the commmand
    for (int i = 1; i <= _command_attempts; i++) {
        try {
            sw::redis::Redis db = _redis_cluster->redis(sv_prefix, false);
            CommandReply reply = db.command(cmd.cbegin(), cmd.cend());
            if (reply.has_error() == 0) {
                _last_prefix = db_prefix;
                return reply;
            }

            // On an error response, print the response and bail
            reply.print_reply_error();
            throw SRRuntimeException(
                "Redis failed to execute command: " + cmd.first_field());
        }
        catch (SmartRedis::Exception& e) {
            // Exception is already prepared, just propagate it
            throw;
        }
        catch (sw::redis::IoError &e) {
            // For an error from Redis, retry unless we're out of chances
            std::string message("Redis IO error when executing command: ");
            message += e.what();
            if (i == _command_attempts) {
                throw SRDatabaseException(message);
            }
            // Else log, retry, and fall through for a retry
            else {
                _context->log_error(LLInfo, message);
            }
        }
        catch (sw::redis::ClosedError &e) {
            // For an error from Redis, retry unless we're out of chances
            std::string message("Redis Closed error when executing command: ");
            message += e.what();
            if (i == _command_attempts) {
                throw SRDatabaseException(message);
            }
            // Else log, retry, and fall through for a retry
            else {
                _context->log_error(LLInfo, message);
            }
        }
        catch (sw::redis::Error &e) {
            // For other errors from Redis, report them immediately
            throw SRRuntimeException(
                std::string("Redis error when executing command: ") +
                e.what());
        }
        catch (std::exception& e) {
            // Should never hit this, so bail immediately if we do
            throw SRInternalException(
                std::string("Unexpected exception executing command: ") +
                e.what());
        }
        catch (...) {
            // Should never hit this, so bail immediately if we do
            throw SRInternalException(
                "Non-standard exception encountered executing command " +
                cmd.first_field());
        }

        // Sleep before the next attempt
        std::this_thread::sleep_for(std::chrono::milliseconds(_command_interval));
    }

    // If we get here, we've run out of retry attempts
    throw SRTimeoutException("Unable to execute command " + cmd.first_field());
}

// Connect to the cluster at the address and port
inline void RedisCluster::_connect(SRAddress& db_address)
{
    // Build a connections object for this connection
    // No need to repeat the build on each connection attempt
    // so we do it outside the loop
    sw::redis::ConnectionOptions connectOpts;
    if (db_address._is_tcp) {
        connectOpts.host = db_address._tcp_host;
        connectOpts.port = db_address._tcp_port;
        connectOpts.type = sw::redis::ConnectionType::TCP;
    }
    else {
        throw SRInternalException(
            "RedisCluster encountered a UDS request in _connect()");
    }
    connectOpts.socket_timeout = std::chrono::milliseconds(
        _DEFAULT_SOCKET_TIMEOUT);

    // Connect
    std::string msg;
    for (int i = 1; i <= _connection_attempts; i++) {
        msg = "Connection attempt " + std::to_string(i) + " of " +
            std::to_string(_connection_attempts);
        _cfgopts->_get_log_context()->log_data(LLDeveloper, msg);

        try {
            // Attempt the connection
            _redis_cluster = new sw::redis::RedisCluster(connectOpts);            return;
        }
        catch (std::bad_alloc& e) {
            // On a memory error, bail immediately
            _redis_cluster = NULL;
            _cfgopts->_get_log_context()->log_data(LLDeveloper, "Memory error");
            throw SRBadAllocException("RedisCluster connection");
        }
        catch (sw::redis::Error& e) {
            // For an error from Redis, retry unless we're out of chances
            msg = "redis error: "; msg += e.what();
            _cfgopts->_get_log_context()->log_data(LLDeveloper, msg);
            _redis_cluster = NULL;
            std::string message("Unable to connect to backend database: ");
            message += e.what();
            if (i == _connection_attempts) {
                throw SRDatabaseException(message);
            }
            // Else log, retry, and fall through for a retry
            else {
                _context->log_error(LLInfo, message);
            }
        }
        catch (std::exception& e) {
            // Should never hit this, so bail immediately if we do
            msg = "std::exception: "; msg += e.what();
            _cfgopts->_get_log_context()->log_data(LLDeveloper, msg);
            _redis_cluster = NULL;
            throw SRInternalException(
                std::string("Unexpected exception while connecting: ") +
                e.what());
        }
        catch (...) {
            // Should never hit this, so bail immediately if we do
            _cfgopts->_get_log_context()->log_data(LLDeveloper, "unknown exception");
            _redis_cluster = NULL;
            throw SRInternalException(
                "A non-standard exception was encountered during client "\
                "connection.");
        }

        // If we get here, the connection attempt failed.
        // Sleep before the next attempt
        _redis_cluster = NULL;
        std::this_thread::sleep_for(std::chrono::milliseconds(_connection_interval));
    }

    // If we get here, we failed to establish a connection
    throw SRTimeoutException(std::string("Connection attempt failed after ") +
                             std::to_string(_connection_attempts) + "tries");
}

// Map the RedisCluster via the CLUSTER SLOTS command
inline void RedisCluster::_map_cluster()
{
    // Clear out our old map
    _db_nodes.clear();
    _address_node_map.clear();

    // Build the CLUSTER SLOTS command
    AddressAnyCommand cmd;
    cmd << "CLUSTER" << "SLOTS";

    // Run it
    CommandReply reply = run(cmd);
    if (reply.has_error() > 0) {
        throw SRRuntimeException("CLUSTER SLOTS command failed");
    }

    // Process results
    _parse_reply_for_slots(reply);
}

// Get the prefix that can be used to address the correct database
// for a given command
std::string RedisCluster::_get_db_node_prefix(Command& cmd)
{
    // Extract the keys from the command
    std::vector<std::string> keys = cmd.get_keys();
    if (keys.size() == 0) {
        throw SRRuntimeException("Command " + cmd.to_string() +
                                 " does not have a key value.");
    }

    // Walk through the keys to find the prefix
    std::string prefix;
    std::vector<std::string>::iterator key_it = keys.begin();
    for ( ; key_it != keys.end(); key_it++) {
        uint16_t db_index = _get_db_node_index(*key_it);
        if (prefix.size() == 0) {
            prefix = _db_nodes[db_index].prefix;
        }
        else if (prefix != _db_nodes[db_index].prefix) {
            throw SRRuntimeException("Multi-key commands are not valid: " +
                                     cmd.to_string());
        }
    }

    // Done
    return prefix;
}

// Get the index in _db_nodes for the provided key
inline uint16_t RedisCluster::_get_db_node_index(const std::string& key)
{
    uint16_t hash_slot = _get_hash_slot(key);
    return  _db_node_hash_search(hash_slot, 0, _db_nodes.size() - 1);
}

// Process the CommandReply for CLUSTER SLOTS to build DBNode information
inline void RedisCluster::_parse_reply_for_slots(CommandReply& reply)
{
    /* Each reply element of the main message, of which there should
    be n_db_nodes, is:
    0) (integer) min slot
    1) (integer) max slot
    2) 0) "ip address"
       1) (integer) port   (note that for clustered Redis, this will always be a TCP address)
       2) "name"
    */
    size_t n_db_nodes = reply.n_elements();
    _db_nodes = std::vector<DBNode>(n_db_nodes);

    for (size_t i = 0; i < n_db_nodes; i++) {
        _db_nodes[i].lower_hash_slot = reply[i][0].integer();
        _db_nodes[i].upper_hash_slot = reply[i][1].integer();
        _db_nodes[i].address._is_tcp = true;
        _db_nodes[i].address._tcp_host = std::string(reply[i][2][0].str(),
                                            reply[i][2][0].str_len());
        _db_nodes[i].address._tcp_port = reply[i][2][1].integer();
        _db_nodes[i].name = std::string(reply[i][2][2].str(),
                                              reply[i][2][2].str_len());
        _db_nodes[i].prefix = _get_crc16_prefix(_db_nodes[i].lower_hash_slot);
        _address_node_map.insert({_db_nodes[i].address.to_string(),
                                  &_db_nodes[i]});
    }

    //Put the vector of db nodes in order based on lower hash slot
    std::sort(_db_nodes.begin(), _db_nodes.end());
}

// Perform inverse CRC16 XOR and shifts
void RedisCluster::_crc_xor_shift(uint64_t& remainder,
                                  const size_t initial_shift,
                                  const size_t n_bits)
{
    uint64_t digit = 1;
    // poly = x^16 + x^12 + x^5 + 1
    uint64_t poly = 69665;

    digit = digit << initial_shift;
    poly = poly << initial_shift;

    for (size_t i = 0; i < n_bits; i++) {
        // Only do the xor if the bit position is 1
        if (remainder & digit) {
            remainder = remainder ^ poly;
        }
        digit = digit << 1;
        poly = poly << 1;
    }
}

// Check if the CRC16 inverse is correct
bool RedisCluster::_is_valid_inverse(uint64_t char_bits,
                                     const size_t n_chars)
{
    uint64_t byte_filter = 255;
    char_bits = char_bits >> 16;

    for(int i = n_chars-1; i >= 0; i--) {
        char c = (char_bits & byte_filter);
        if (c == '}' || c == '{') {
            return false;
        }
        char_bits = char_bits >> 8;
    }
    return true;
}

// Get a DBNode prefix for the provided hash slot
std::string RedisCluster::_get_crc16_prefix(uint64_t hash_slot)
{

    if (hash_slot > 16384) {
        throw SRRuntimeException("Hash slot " + std::to_string(hash_slot) +
                                 " is beyond the limit of 16384.  A "\
                                 "prefix cannot be generated by "\
                                 "_get_crc16_prefix().");
    }

    /*
       The total number of XOR shifts is a minimum of 16
       (2 chars).  This shift is done first and then subsequent
       shifts are performed if the prefix contains forbidden characters.
    */
    uint64_t bit_shifts = 16;
    uint64_t n_chars = 2;

    _crc_xor_shift(hash_slot, 0, bit_shifts);

    /*
       Continue inverse XOR shifts until a valid prefix is constructed.
       Empirically we know that no more than 8 additional operations
       are required for 16384 hash slots, so an error is thrown
       for more than 8 shifts so we don't have an infinite loop programmed.
    */
    while (!_is_valid_inverse(hash_slot, n_chars)) {
        if (bit_shifts > 24) {
            throw SRRuntimeException("The maximum bit shifts were"\
                                     "exceeded in the CRC16 inverse "\
                                     "calculation.");
        }
        _crc_xor_shift(hash_slot, bit_shifts, 1);
        bit_shifts++;
        n_chars = (bit_shifts + 7) / 8;
    }

    // Turn the inverse bits into a prefix
    std::string prefix(n_chars, 0);

    uint64_t byte_filter = 255;
    hash_slot = hash_slot >> 16;
    for(int i = n_chars - 1; i >= 0; i--) {
        prefix[i] = (hash_slot & byte_filter);
        hash_slot = hash_slot >> 8;
    }

    return prefix;
}

// Determine if the key has a substring enclosed by "{" and "}" characters
bool RedisCluster::_has_hash_tag(const std::string& key)
{
    size_t first = key.find('{');
    size_t second = key.find('}');
    return (first != std::string::npos && second != std::string::npos &&
            second > first);
}

// Return the key enclosed by "{" and "}" characters
std::string RedisCluster::_get_hash_tag(const std::string& key)
{
    // If no hash tag, bail
    if (!_has_hash_tag(key))
        return key;

    // Extract the hash tag
    size_t first = key.find('{');
    size_t second = key.find('}');
    return key.substr(first + 1, second - first - 1);
}

// Get the hash slot for a key
uint16_t RedisCluster::_get_hash_slot(const std::string& key)
{
    std::string hash_key = _get_hash_tag(key);
    return sw::redis::crc16(hash_key.c_str(), hash_key.size()) % 16384;
}

// Get the index of the DBNode responsible for the hash slot
uint16_t RedisCluster::_db_node_hash_search(uint16_t hash_slot,
                                            unsigned lhs,
                                            unsigned rhs)
{
    // Find the DBNode via binary search
    uint16_t m = (lhs + rhs) / 2;

    // If this is the correct slot, we're done
    if (_db_nodes[m].lower_hash_slot <= hash_slot &&
        _db_nodes[m].upper_hash_slot >= hash_slot) {
        return m;
    }

    // Otherwise search in the appropriate half
    else {
        if (_db_nodes[m].lower_hash_slot > hash_slot)
            return _db_node_hash_search(hash_slot, lhs, m - 1);
        else
            return _db_node_hash_search(hash_slot, m + 1, rhs);
    }
}

// Attaches a prefix and constant suffix to keys to enforce identical
//  hash slot constraint
std::vector<std::string>
RedisCluster::_get_tmp_names(std::vector<std::string> names,
                             std::string db_prefix)
{
    std::vector<std::string> tmp;
    std::vector<std::string>::iterator it = names.begin();
    for ( ; it != names.end(); it++) {
        std::string new_key = "{" + db_prefix + "}." + *it + ".TMP";
        tmp.push_back(new_key);
    }
    return tmp;
}

// Delete multiple keys (assumesthat all keys use the same hash slot)
void RedisCluster::_delete_keys(std::vector<std::string> keys)
{
    // Build the command
    MultiKeyCommand cmd;
    cmd << "DEL";
    cmd.add_keys(keys);

    // Run it, ignoring failure
    (void)run(cmd);
}

// Retrieve the optimum model prefix for the set of inputs
DBNode* RedisCluster::_get_model_script_db(const std::string& name,
                                           std::vector<std::string>& inputs,
                                           std::vector<std::string>& outputs)
{
    /* This function calculates the optimal model name to use
    to run the provided inputs.  If a cluster is not being used,
    the model name is returned, else a prefixed model name is returned.
    */

    // TODO we should randomly choose the max if there are multiple maxes

    std::vector<int> hash_slot_tally(_db_nodes.size(), 0);

    for (size_t i = 0; i < inputs.size(); i++) {
        uint16_t hash_slot = _get_hash_slot(inputs[i]);
        uint16_t db_index = _db_node_hash_search(hash_slot, 0, _db_nodes.size());
        hash_slot_tally[db_index]++;
    }

    for (size_t i = 0; i < outputs.size(); i++) {
        uint16_t hash_slot = _get_hash_slot(outputs[i]);
        uint16_t db_index = _db_node_hash_search(hash_slot, 0, _db_nodes.size());
        hash_slot_tally[db_index]++;
    }

    // Determine which DBNode has the most hashes
    int max_hash = -1;
    DBNode* db = NULL;
    for (size_t i = 0; i < _db_nodes.size(); i++) {
        if (hash_slot_tally[i] > max_hash) {
            max_hash = hash_slot_tally[i];
            db = &(_db_nodes[i]);
        }
    }
    return db;
}

// Run a CommandList via a Pipeline
PipelineReply RedisCluster::run_in_pipeline(CommandList& cmdlist)
{
    // Convert from CommandList to vector and grab the shard along
    // the way
    std::vector<Command*> cmds;
    std::string shard_prefix = _db_nodes[0].prefix;
    bool shard_found = false;
    for (auto it = cmdlist.begin(); it != cmdlist.end(); ++it) {
        cmds.push_back(*it);
        if (!shard_found && (*it)->has_keys()) {
            shard_prefix = _get_db_node_prefix(*(*it));
            shard_found = true;
        }
    }

    // Run the commands
    return _run_pipeline(cmds, shard_prefix);
}

// Build and run unordered pipeline
PipelineReply RedisCluster::_run_pipeline(
    std::vector<Command*>& cmds,
    std::string& shard_prefix)
{
    PipelineReply reply;
    for (int i = 1; i <= _command_attempts; i++) {
        try {
            // Get pipeline object for shard (no new connection)
            sw::redis::Pipeline pipeline =
                _redis_cluster->pipeline(shard_prefix, false);

            // Loop over all commands and add to the pipeline
            for (size_t i = 0; i < cmds.size(); i++) {
                // Add the commands to the pipeline
                pipeline.command(cmds[i]->cbegin(), cmds[i]->cend());
            }

            // Execute the pipeline
            reply = pipeline.exec();

            // Check the replies
            if (reply.has_error()) {
                throw SRRuntimeException("Redis failed to execute the pipeline");
            }

            // If we get here, it all worked
            return reply;
        }
        catch (SmartRedis::Exception& e) {
            // Exception is already prepared, just propagate it
            throw;
        }
        catch (sw::redis::IoError &e) {
            // For an error from Redis, retry unless we're out of chances
            if (i == _command_attempts) {
                throw SRDatabaseException(
                    std::string("Redis IO error when executing the pipeline: ") +
                    e.what());
            }
            // else, Fall through for a retry
        }
        catch (sw::redis::ClosedError &e) {
            // For an error from Redis, retry unless we're out of chances
            if (i == _command_attempts) {
                throw SRDatabaseException(
                    std::string("Redis Closed error when executing the "\
                                "pipeline: ") + e.what());
            }
            // else, Fall through for a retry
        }
        catch (sw::redis::Error &e) {
            // For other errors from Redis, report them immediately
            throw SRRuntimeException(
                std::string("Redis error when executing the pipeline: ") +
                    e.what());
        }
        catch (std::exception& e) {
            // Should never hit this, so bail immediately if we do
            throw SRInternalException(
                std::string("Unexpected exception executing the pipeline: ") +
                    e.what());
        }
        catch (...) {
            // Should never hit this, so bail immediately if we do
            throw SRInternalException(
                "Non-standard exception encountered executing the pipeline");
        }

        // Sleep before the next attempt
        std::this_thread::sleep_for(std::chrono::milliseconds(_command_interval));
    }

    // If we get here, we've run out of retry attempts
    throw SRTimeoutException("Unable to execute pipeline");
}

// Create a string representation of the Redis connection
std::string RedisCluster::to_string() const
{
    std::string result("Clustered Redis connection:\n");
    result += RedisServer::to_string();
    return result;
}
