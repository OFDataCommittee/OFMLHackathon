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

#include "redis.h"
#include "srexception.h"

using namespace SmartRedis;

// Redis constructor.
Redis::Redis() : RedisServer()
{
    std::string address_port = _get_ssdb();
    _add_to_address_map(address_port);
    _connect(address_port);
}

// Redis constructor. Uses address provided to constructor instead of environment variables
Redis::Redis(std::string address_port) : RedisServer()
{
    _add_to_address_map(address_port);
    _connect(address_port);
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
    if (not is_addressable(cmd.get_address(), cmd.get_port()))
        throw SRRuntimeException("The provided host and port do not match "\
                                 "the host and port used to initialize the "\
                                 "non-cluster client connection.");
    return this->_run(cmd);
}

// Run an address-any Command on the server
CommandReply Redis::run(AddressAnyCommand& cmd){
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
    cmd.add_field("EXISTS");
    cmd.add_field(key);

    // Run it
    CommandReply reply = run(cmd);
    if (reply.has_error() > 0)
        throw SRRuntimeException("Error encountered while checking "\
                                 "for existence of key " + key);
    return (bool)reply.integer();
}

// Check if address is valid
bool Redis::is_addressable(const std::string& address,
                           const uint64_t& port)
{
    return _address_node_map.find(address + ":" + std::to_string(port)) !=
        _address_node_map.end();
}

// Put a Tensor on the server
CommandReply Redis::put_tensor(TensorBase& tensor)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd.add_field("AI.TENSORSET");
    cmd.add_field(tensor.name());
    cmd.add_field(tensor.type_str());
    cmd.add_fields(tensor.dims());
    cmd.add_field("BLOB");
    cmd.add_field_ptr(tensor.buf());

    // Run it
    return run(cmd);
}

// Get a Tensor from the server
CommandReply Redis::get_tensor(const std::string& key)
{
    // Build the command
    GetTensorCommand cmd;
    cmd.add_field("AI.TENSORGET");
    cmd.add_field(key);
    cmd.add_field("META");
    cmd.add_field("BLOB");

    // Run it
    return run(cmd);
}

// Rename a tensor in the database
CommandReply Redis::rename_tensor(const std::string& key,
                                  const std::string& new_key)
{
    // Build the command
    MultiKeyCommand cmd;
    cmd.add_field("RENAME");
    cmd.add_field(key);
    cmd.add_field(new_key);

    // Run it
    return run(cmd);
}

// Delete a tensor in the database
CommandReply Redis::delete_tensor(const std::string& key)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd.add_field("DEL");
    cmd.add_field(key, true);

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
    cmd_get.add_field("AI.TENSORGET");
    cmd_get.add_field(src_key, true);
    cmd_get.add_field("META");
    cmd_get.add_field("BLOB");

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
    cmd_put.add_field("AI.TENSORSET");
    cmd_put.add_field(dest_key, true);
    cmd_put.add_field(TENSOR_STR_MAP.at(type));
    cmd_put.add_fields(dims);
    cmd_put.add_field("BLOB");
    cmd_put.add_field_ptr(blob);

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
                              std::string_view model,
                              const std::string& backend,
                              const std::string& device,
                              int batch_size,
                              int min_batch_size,
                              const std::string& tag,
                              const std::vector<std::string>& inputs,
                              const std::vector<std::string>& outputs
                              )
{
    // Build the command
    SingleKeyCommand cmd;
    cmd.add_field("AI.MODELSET");
    cmd.add_field(model_name);
    cmd.add_field(backend);
    cmd.add_field(device);

    // Add optional fields if requested
    if (tag.size() > 0) {
        cmd.add_field("TAG");
        cmd.add_field(tag);
    }
    if (batch_size > 0) {
        cmd.add_field("BATCHSIZE");
        cmd.add_field(std::to_string(batch_size));
    }
    if (min_batch_size > 0) {
        cmd.add_field("MINBATCHSIZE");
        cmd.add_field(std::to_string(min_batch_size));
    }
    if (inputs.size() > 0) {
        cmd.add_field("INPUTS");
        cmd.add_fields(inputs);
    }
    if (outputs.size() > 0) {
        cmd.add_field("OUTPUTS");
        cmd.add_fields(outputs);
    }
    cmd.add_field("BLOB");
    cmd.add_field_ptr(model);

    // Run it
    return run(cmd);
}

// Set a script from a string_view buffer in the database for future execution
CommandReply Redis::set_script(const std::string& key,
                               const std::string& device,
                               std::string_view script)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd.add_field("AI.SCRIPTSET");
    cmd.add_field(key, true);
    cmd.add_field(device);
    cmd.add_field("SOURCE");
    cmd.add_field_ptr(script);

    // Run it
    return run(cmd);
}

// Run a model in the database using the specificed input and output tensors
CommandReply Redis::run_model(const std::string& key,
                              std::vector<std::string> inputs,
                              std::vector<std::string> outputs)
{
    // Build the command
    CompoundCommand cmd;
    cmd.add_field("AI.MODELRUN");
    cmd.add_field(key);
    cmd.add_field("INPUTS");
    cmd.add_fields(inputs);
    cmd.add_field("OUTPUTS");
    cmd.add_fields(outputs);

    // Run it
    return run(cmd);
}

// Run a script function in the database using the specificed input and
// output tensors
CommandReply Redis::run_script(const std::string& key,
                              const std::string& function,
                              std::vector<std::string> inputs,
                              std::vector<std::string> outputs)
{
    // Build the command
    CompoundCommand cmd;
    cmd.add_field("AI.SCRIPTRUN");
    cmd.add_field(key);
    cmd.add_field(function);
    cmd.add_field("INPUTS");
    cmd.add_fields(inputs);
    cmd.add_field("OUTPUTS");
    cmd.add_fields(outputs);

    // Run it
    return run(cmd);
}

// Retrieve the model from the database
CommandReply Redis::get_model(const std::string& key)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd.add_field("AI.MODELGET");
    cmd.add_field(key);
    cmd.add_field("BLOB");

    // Run it
    return run(cmd);
}

// Retrieve the script from the database
CommandReply Redis::get_script(const std::string& key)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd.add_field("AI.SCRIPTGET");
    cmd.add_field(key, true);
    cmd.add_field("SOURCE");

    // Run it
    return run(cmd);
}

inline CommandReply Redis::_run(const Command& cmd)
{
    CommandReply reply;
    bool executed = false;
    for (int i = 1; i <= _command_attempts; i++) {
        try {
            reply = _redis->command(cmd.cbegin(), cmd.cend());
            if (reply.has_error() == 0)
                return reply;
            executed = true;
            break;
        }
        catch (sw::redis::IoError &e) {
            // Fall through for a retry
        }
        catch (sw::redis::ClosedError &e) {
            // Fall through for a retry
        }
        catch (std::exception& e) {
            throw SRRuntimeException(e.what());
        }
        catch (...) {
            throw SRInternalException("Non-standard exception "\
                                      "encountered during command " +
                                      cmd.first_field() + " execution. ");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(_command_interval));
    }

    // If we get here, we've either run out of retry attempts or gotten
    // a failure back from the database
    if (executed) {
        if (reply.has_error() > 0)
            reply.print_reply_error();
        throw SRRuntimeException("Redis failed to execute command: " +
                                 cmd.first_field());
    }

    // Since we didn't execute, we must have run out of retry attempts
    throw SRTimeoutException("Unable to execute command" + cmd.first_field());
}

inline void Redis::_add_to_address_map(std::string address_port)
{
    if (address_port.rfind("tcp://", 0) == 0)
        address_port = address_port.substr(6, std::string::npos);
    else if (address_port.rfind("unix://", 0) == 0)
        address_port = address_port.substr(7, std::string::npos);

    _address_node_map.insert({address_port, nullptr});
}

inline void Redis::_connect(std::string address_port)
{
    // Try to create the sw::redis::Redis object
    try {
        _redis = new sw::redis::Redis(address_port);
    }
    catch (sw::redis::Error& e) {
        _redis = NULL;
        throw SRDatabaseException(std::string("Unable to connect to "\
                                  "backend Redis database: ") +
                                  e.what());
    }
    catch (std::bad_alloc& e) {
        throw SRBadAllocException("Redis connection");
    }
    catch (std::exception& e) {
        _redis = NULL;
        throw SRRuntimeException("Failed to create Redis object with error: " +
                                 std::string(e.what()));
    }
    catch (...) {
        _redis = NULL;
        throw SRInternalException("A non-standard exception encountered "\
                                  "during client connection.");
    }

    // Attempt to have the sw::redis::Redis object
    // make a connection using the PING command
    for (int i = 1; i <= _connection_attempts; i++) {
        try {
            if (_redis->ping().compare("PONG") == 0) {
                return;
            }
            else if (i == _connection_attempts) {
                throw SRTimeoutException(std::string("Connection attempt "\
                                         "failed after ") +
                                         std::to_string(i) + "tries");
            }
        }
        catch (std::exception& e) {
            if (i == _connection_attempts) {
                throw SRRuntimeException(e.what());
            }
        }
        catch (...) {
            if (i == _connection_attempts) {
                throw SRRuntimeException("A non-standard exception encountered"\
                                         " during client connection.");
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(_connection_interval));
    }

    // Should never get here
    throw SRInternalException("End of _connect reached unexpectedly");
}
