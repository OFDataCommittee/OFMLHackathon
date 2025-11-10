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
#include "redis.h"
#include "srexception.h"
#include "utility.h"
#include "srobject.h"

using namespace SmartRedis;

// Redis constructor.
Redis::Redis(ConfigOptions* cfgopts)
    : RedisServer(cfgopts)
{
    SRAddress db_address(_get_ssdb());
    // Remember whether it's a unix domain socket for later
    _is_domain_socket = !db_address._is_tcp;
    _add_to_address_map(db_address);
    _connect(db_address);
}

// Redis constructor. Uses address provided to constructor instead of environment variables
Redis::Redis(ConfigOptions* cfgopts, std::string addr_spec)
    : RedisServer(cfgopts)
{
    SRAddress db_address(addr_spec);
    _add_to_address_map(db_address);
    _connect(db_address);
}

// Redis destructor
Redis::~Redis()
{
    if (_redis != NULL) {
        delete _redis;
        _redis = NULL;
    }
}

// Run a single-key Command on the server
CommandReply Redis::run(SingleKeyCommand& cmd){
    return _run(cmd);
}

// Run a multi-key Command on the server
CommandReply Redis::run(MultiKeyCommand& cmd){
    return _run(cmd);
}

// Run a compound Command on the server
CommandReply Redis::run(CompoundCommand& cmd){
    return _run(cmd);
}

// Run an address-at Command on the server
CommandReply Redis::run(AddressAtCommand& cmd){
    if (!is_addressable(cmd.get_address())) {
        std::string msg("The provided address does not match "\
                        "the address used to initialize the "\
                        "non-cluster client connection.");
        msg += " Received: " + cmd.get_address().to_string();
        throw SRRuntimeException(msg);
    }
    return this->_run(cmd);
}

// Run an address-any Command on the server
CommandReply Redis::run(AddressAnyCommand& cmd){
    return _run(cmd);
}

// Run an address-all Command on the server
CommandReply Redis::run(AddressAllCommand& cmd){
    return _run(cmd);
}

// Run a Command list on the server
std::vector<CommandReply> Redis::run(CommandList& cmds)
{
    std::vector<CommandReply> replies;
    CommandList::iterator cmd = cmds.begin();
    for ( ; cmd != cmds.end(); cmd++) {
        replies.push_back(dynamic_cast<Command*>(*cmd)->run_me(this));
    }
    return replies;
}

PipelineReply
Redis::run_via_unordered_pipelines(CommandList& cmd_list)
{
    // Get pipeline object (no new connection)
    sw::redis::Pipeline pipeline = _redis->pipeline(false);

    // Iterator over all commands and add to pipeline
    CommandList::iterator cmd = cmd_list.begin();

    for ( ; cmd != cmd_list.end(); cmd++) {
        pipeline.command((*cmd)->cbegin(), (*cmd)->cend());
    }

    // Execute the pipeline and return value
    return PipelineReply(pipeline.exec());
}

// Check if a model or script key exists in the database
bool Redis::model_key_exists(const std::string& key)
{
    return key_exists(key);
}

// Check if a key exists in the database
bool Redis::key_exists(const std::string& key)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd << "EXISTS" << key;

    // Run it
    CommandReply reply = run(cmd);
    if (reply.has_error() > 0)
        throw SRRuntimeException("Error encountered while checking "\
                                 "for existence of key " + key);
    return (bool)reply.integer();
}

// Check if a hash field exists in the database
bool Redis::hash_field_exists(const std::string& key,
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

// Check if address is valid
bool Redis::is_addressable(const SRAddress& address) const
{
    return _address_node_map.find(address.to_string()) !=
        _address_node_map.end();
}

// Put a Tensor on the server
CommandReply Redis::put_tensor(TensorBase& tensor)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd << "AI.TENSORSET" << Keyfield(tensor.name()) << tensor.type_str()
        << tensor.dims() << "BLOB" << tensor.buf();

    // Run it
    return run(cmd);
}

// Get a Tensor from the server
CommandReply Redis::get_tensor(const std::string& key)
{
    // Build the command
    GetTensorCommand cmd;
    cmd << "AI.TENSORGET" << Keyfield(key) << "META" << "BLOB";

    // Run it
    return run(cmd);
}

// Get a list of Tensor from the server
PipelineReply Redis::get_tensors(const std::vector<std::string>& keys)
{
    // Build up the commands to get the tensors
    CommandList cmdlist; // This just holds the memory
    std::vector<Command*> cmds;
    for (auto it = keys.begin(); it != keys.end(); ++it) {
        GetTensorCommand* cmd = cmdlist.add_command<GetTensorCommand>();
        (*cmd) << "AI.TENSORGET" << Keyfield(*it) << "META" << "BLOB";
        cmds.push_back(cmd);
    }

    // Run them via pipeline
    return _run_pipeline(cmds);
}

// Rename a tensor in the database
CommandReply Redis::rename_tensor(const std::string& key,
                                  const std::string& new_key)
{
    // Build the command
    MultiKeyCommand cmd;
    cmd << "RENAME" << Keyfield(key) << Keyfield(new_key);

    // Run it
    return run(cmd);
}

// Delete a tensor in the database
CommandReply Redis::delete_tensor(const std::string& key)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd << "DEL" << Keyfield(key);

    // Run it
    return run(cmd);
}

// Copy a tensor from the source key to the destination key
CommandReply Redis::copy_tensor(const std::string& src_key,
                                const std::string& dest_key)
{
    //TODO can we do COPY for same hash slot or database?

    // Build a GET command to fetch the tensor
    GetTensorCommand cmd_get;
    cmd_get << "AI.TENSORGET" << Keyfield(src_key) << "META" << "BLOB";

    // Run the GET command
    CommandReply cmd_get_reply = run(cmd_get);
    if (cmd_get_reply.has_error() > 0) {
        throw SRRuntimeException("Failed to retrieve tensor " +
                                 src_key + "from database");
    }

    // Decode the tensor
    std::vector<size_t> dims = cmd_get.get_dims(cmd_get_reply);
    std::string_view blob = cmd_get.get_data_blob(cmd_get_reply);
    SRTensorType type = cmd_get.get_data_type(cmd_get_reply);

    // Build a PUT command to send the tensor back to the database
    // under the new key
    MultiKeyCommand cmd_put;
    cmd_put << "AI.TENSORSET" << Keyfield(dest_key) << TENSOR_STR_MAP.at(type)
            << dims << "BLOB" << blob;

    // Run the PUT command
    return run(cmd_put);
}

// Copy a vector of tensors from source keys to destination keys
CommandReply Redis::copy_tensors(const std::vector<std::string>& src,
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

// Set a model from std::string_view buffer in the database for future execution
CommandReply Redis::set_model(const std::string& model_name,
                              const std::vector<std::string_view>& model,
                              const std::string& backend,
                              const std::string& device,
                              int batch_size,
                              int min_batch_size,
                              int min_batch_timeout,
                              const std::string& tag,
                              const std::vector<std::string>& inputs,
                              const std::vector<std::string>& outputs
                              )
{
    // Build the command
    SingleKeyCommand cmd;
    cmd << "AI.MODELSTORE" << Keyfield(model_name) << backend << device;

    // Add optional fields if requested
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
    if (inputs.size() > 0) {
        cmd << "INPUTS" << std::to_string(inputs.size()) <<  inputs;
    }
    if (outputs.size() > 0) {
        cmd << "OUTPUTS" << std::to_string(outputs.size()) << outputs;
    }
    cmd << "BLOB" << model;

    // Run it
    return run(cmd);
}

// Set a model from std::string_view buffer in the
// database for future execution in a multi-GPU system
void Redis::set_model_multigpu(const std::string& name,
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
    CommandReply result;
    for (int i = first_gpu; i < num_gpus; i++) {
        std::string device = "GPU:" + std::to_string(i);
        std::string model_key = name + "." + device;
        result = set_model(
            model_key, model, backend, device, batch_size, min_batch_size, min_batch_timeout,
            tag, inputs, outputs);
        if (result.has_error() > 0) {
            throw SRRuntimeException("Failed to set model for GPU " + std::to_string(i));
        }
    }

    // Add a version for get_model to find
    result = set_model(
        name, model, backend, "GPU", batch_size, min_batch_size, min_batch_timeout,
        tag, inputs, outputs);
    if (result.has_error() > 0) {
        throw SRRuntimeException("Failed to set general model");
    }
}

// Set a script from a string_view buffer in the database for future execution
CommandReply Redis::set_script(const std::string& key,
                               const std::string& device,
                               std::string_view script)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd << "AI.SCRIPTSET" << Keyfield(key) << device << "SOURCE" << script;

    // Run it
    return run(cmd);
}

// Set a script from std::string_view buffer in the
// database for future execution in a multi-GPU system
void Redis::set_script_multigpu(const std::string& name,
                                const std::string_view& script,
                                int first_gpu,
                                int num_gpus)
{
    // Store a copy of the script for each GPU
    CommandReply result;
    for (int i = first_gpu; i < num_gpus; i++) {
        std::string device = "GPU:" + std::to_string(i);
        std::string script_key = name + "." + device;
        result = set_script(script_key, device, script);
        if (result.has_error() > 0) {
            throw SRRuntimeException("Failed to set script for GPU " + std::to_string(i));
        }
    }

    // Add a copy for get_script to find
    result = set_script(name, "GPU", script);
    if (result.has_error() > 0) {
        throw SRRuntimeException("Failed to set general script");
    }
}

// Run a model in the database using the specified input and output tensors
CommandReply Redis::run_model(const std::string& key,
                              std::vector<std::string> inputs,
                              std::vector<std::string> outputs)
{
    // Check for a non-default timeout setting
    int run_timeout;
    get_config_integer(run_timeout, _MODEL_TIMEOUT_ENV_VAR,
                       _DEFAULT_MODEL_TIMEOUT);

    // Build the command
    CompoundCommand cmd;
    cmd << "AI.MODELEXECUTE" << Keyfield(key)
        << "INPUTS" << std::to_string(inputs.size()) << inputs
        << "OUTPUTS" << std::to_string(outputs.size()) << outputs
        << "TIMEOUT" << std::to_string(run_timeout);

    // Run it
    return run(cmd);
}

// Run a model in the database using the
// specified input and output tensors in a multi-GPU system
void Redis::run_model_multigpu(const std::string& name,
                               std::vector<std::string> inputs,
                               std::vector<std::string> outputs,
                               int offset,
                               int first_gpu,
                               int num_gpus)
{
    int gpu = first_gpu + _modulo(offset, num_gpus);
    std::string device = "GPU:" + std::to_string(gpu);
    CommandReply result = run_model(name + "." + device, inputs, outputs);
    if (result.has_error() > 0) {
        throw SRRuntimeException(
            "An error occured while executing the model on " + device);
    }
}

// Run a script function in the database using the specified input and
// output tensors
CommandReply Redis::run_script(const std::string& key,
                              const std::string& function,
                              std::vector<std::string> inputs,
                              std::vector<std::string> outputs)
{
    // Build the command
    CompoundCommand cmd;
    cmd << "AI.SCRIPTRUN" << Keyfield(key) << function
        << "INPUTS" << inputs << "OUTPUTS" << outputs;

    // Run it
    return run(cmd);
}

// Use multiple GPUs to run a script function in the database using the
// specified input and output tensors
void Redis::run_script_multigpu(const std::string& name,
                                const std::string& function,
                                std::vector<std::string>& inputs,
                                std::vector<std::string>& outputs,
                                int offset,
                                int first_gpu,
                                int num_gpus)
{
    int gpu = first_gpu + _modulo(offset, num_gpus);
    std::string device = "GPU:" + std::to_string(gpu);
    CommandReply result = run_script(
        name + "." + device, function, inputs, outputs);
    if (result.has_error() > 0) {
        throw SRRuntimeException(
            "An error occured while executing the script on " + device);
    }
}

// Delete a model from the database
CommandReply Redis::delete_model(const std::string& key)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd << "AI.MODELDEL" << Keyfield(key);

    // Run it
    return run(cmd);
}

// Remove a model from the database that was stored
// for use with multiple GPUs
void Redis::delete_model_multigpu(
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
CommandReply Redis::delete_script(const std::string& key)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd << "AI.SCRIPTDEL" << Keyfield(key);

    // Run it
    return run(cmd);
}

// Remove a script from the database that was stored
// for use with multiple GPUs
void Redis::delete_script_multigpu(
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
CommandReply Redis::get_model(const std::string& key)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd << "AI.MODELGET" << Keyfield(key) << "BLOB";

    // Run it
    return run(cmd);
}

// Retrieve the script from the database
CommandReply Redis::get_script(const std::string& key)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd << "AI.SCRIPTGET" << Keyfield(key) << "SOURCE";

    // Run it
    return run(cmd);
}

// Retrieve the model and script AI.INFO
CommandReply Redis::get_model_script_ai_info(const std::string& address,
                                             const std::string& key,
                                             const bool reset_stat)
{
    AddressAtCommand cmd;
    SRAddress db_address(address);

    // Determine the prefix we need for the model or script
    if (!is_addressable(db_address)) {
        throw SRRuntimeException("The provided address does not match "\
                                 "the address used to initialize the "\
                                 "non-cluster client connection.");
    }

    // Build the Command
    cmd.set_exec_address(db_address);
    cmd << "AI.INFO" << Keyfield(key);

    // Optionally add RESETSTAT to the command
    if (reset_stat) {
        cmd << "RESETSTAT";
    }

    return run(cmd);
}

// Retrieve the current model chunk size
int Redis::get_model_chunk_size()
{
    // If we've already set a chunk size, just return it
    if (_model_chunk_size != _UNKNOWN_MODEL_CHUNK_SIZE)
        return _model_chunk_size;

    // Build the command
    AddressAnyCommand cmd;
    cmd << "AI.CONFIG" << "GET" << "MODEL_CHUNK_SIZE";

    CommandReply reply = _run(cmd);
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
void Redis::set_model_chunk_size(int chunk_size)
{
    AddressAnyCommand cmd;
    cmd << "AI.CONFIG" << "MODEL_CHUNK_SIZE" << std::to_string(chunk_size);

    CommandReply reply = _run(cmd);
    if (reply.has_error() > 0)
        throw SRRuntimeException("AI.CONFIG MODEL_CHUNK_SIZE command failed");

    // Store the new model chunk size for later
    _model_chunk_size = chunk_size;
}

inline CommandReply Redis::_run(const Command& cmd)
{
    for (int i = 1; i <= _command_attempts; i++) {
        try {
            // Run the command
            CommandReply reply = _redis->command(cmd.cbegin(), cmd.cend());
            if (reply.has_error() == 0)
                return reply;

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

        // If we get here, the execution attempt failed.
        // Sleep before the next attempt
        std::this_thread::sleep_for(std::chrono::milliseconds(_command_interval));
    }

    // If we get here, we've run out of retry attempts
    throw SRTimeoutException("Unable to execute command" + cmd.first_field());
}

inline void Redis::_add_to_address_map(SRAddress& db_address)
{
    _address_node_map.insert({db_address.to_string(), nullptr});
}

inline void Redis::_connect(SRAddress& db_address)
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
        connectOpts.path = db_address._uds_file;
        connectOpts.type = sw::redis::ConnectionType::UNIX;
    }
    connectOpts.socket_timeout = std::chrono::milliseconds(
        _DEFAULT_SOCKET_TIMEOUT);

    // Connect
    for (int i = 1; i <= _connection_attempts; i++) {
        try {
            // Try to create the sw::redis::Redis object
            _redis = new sw::redis::Redis(connectOpts);

            // Attempt to have the sw::redis::Redis object
            // make a connection using the PING command
            if (_redis->ping().compare("PONG") == 0) {
                return;
            }
        }
        catch (std::bad_alloc& e) {
            // On a memory error, bail immediately
            if (_redis != NULL) {
                delete _redis;
                _redis = NULL;
            }
            throw SRBadAllocException("Redis connection");
        }
        catch (sw::redis::Error& e) {
            // For an error from Redis, retry unless we're out of chances
            if (_redis != NULL) {
                delete _redis;
                _redis = NULL;
            }
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
            if (_redis != NULL) {
                delete _redis;
                _redis = NULL;
            }
            throw SRInternalException(
                std::string("Unexpected exception while connecting: ") +
                e.what());
        }
        catch (...) {
            // Should never hit this, so bail immediately if we do
            if (_redis != NULL) {
                delete _redis;
                _redis = NULL;
            }
            throw SRInternalException(
                "A non-standard exception encountered "\
                "during client connection.");
        }

        // Delay before the retry
        if (_redis != NULL) {
            delete _redis;
            _redis = NULL;
        }
        if (i < _connection_attempts) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(_connection_interval));
        }
    }

    // If we get here, we've run out of retry attempts
    throw SRTimeoutException(std::string("Connection attempt failed after ") +
                             std::to_string(_connection_attempts) + "tries");
}

// Run a CommandList via a Pipeline
PipelineReply Redis::run_in_pipeline(CommandList& cmdlist)
{
    // Convert from CommandList to vector
    std::vector<Command*> cmds;
    for (auto it = cmdlist.begin(); it != cmdlist.end(); ++it) {
        cmds.push_back(*it);
    }

    // Run the commands
    return _run_pipeline(cmds);
}

// Build and run unordered pipeline
PipelineReply Redis::_run_pipeline(std::vector<Command*>& cmds)
{
    PipelineReply reply;
    for (int i = 1; i <= _command_attempts; i++) {
        try {
            // Get pipeline object for shard (no new connection)
            auto pipeline = _redis->pipeline(false);

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
std::string Redis::to_string() const
{
    std::string result("Non-clustered Redis connection:\n");
    result += RedisServer::to_string();
    return result;
}
