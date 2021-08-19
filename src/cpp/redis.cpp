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

using namespace SmartRedis;

// Redis constructor.
Redis::Redis() : RedisServer()
{
    std::string address_port = this->_get_ssdb();
    this->_connect(address_port);
}

// Redis constructor. Uses address provided to constructor instead of environment variables
Redis::Redis(std::string address_port) : RedisServer()
{
    this->_connect(address_port);
}

// Redis destructor
Redis::~Redis()
{
    if (this->_redis != NULL) {
        delete this->_redis;
        this->_redis = NULL;
    }
}

// Run a single-key or single-hash slot Command on the server
CommandReply Redis::run(Command& cmd)
{
    CommandReply reply;
    bool do_sleep = false;
    for (int n_trial = 0; n_trial < 100; n_trial++) {
        do_sleep = false;
        try {
            Command::iterator cmd_fields_start = cmd.begin();
            Command::iterator cmd_fields_end = cmd.end();
            reply = this->_redis->command(cmd_fields_start, cmd_fields_end);

            if (reply.has_error() == 0)
                return reply;
            break;
        }
        catch (sw::redis::IoError &e) {
            do_sleep = true;
        }
        catch (sw::redis::ClosedError &e) {
            do_sleep = true;
        }
        catch (std::exception& e) {
            throw std::runtime_error(e.what());
        }
        catch (...) {
            throw std::runtime_error("A non-standard exception "\
                                     "encountered during command " +
                                     cmd.first_field() +
                                     " execution. ");
        }

        if (do_sleep) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

    if (reply.has_error() > 0)
        reply.print_reply_error();
    throw std::runtime_error("Redis failed to execute command: " +
                                cmd.first_field());

    return reply;
}

// Run multiple single-key or single-hash slot Command on the server.
// Each Command in the CommandList is run sequentially
CommandReply Redis::run(CommandList& cmds)
{
    CommandList::iterator cmd = cmds.begin();
    CommandReply reply;
    int error_count = 0;
    for ( ; cmd != cmds.end(); cmd++) {
        reply = this->run(**cmd);
        error_count += reply.has_error();
    }
    return reply;
}

// Check if a model or script key exists in the database
bool Redis::model_key_exists(const std::string& key)
{
    return this->key_exists(key);
}

// Check if a key exists in the database
bool Redis::key_exists(const std::string& key)
{
    // Build the command
    Command cmd;
    cmd.add_field("EXISTS");
    cmd.add_field(key);

    // Run it
    CommandReply reply = this->run(cmd);
    if (reply.has_error() > 0)
        throw std::runtime_error("Error encountered while checking "\
                                 "for existence of key " + key);
    return (bool)reply.integer();
}

// Check if address is valid
bool Redis::is_addressable(const std::string& address,
                           const uint64_t& port)
{
    return this->_address_node_map.find(address + ":"
                        + std::to_string(port))
                        != this->_address_node_map.end();
}

// Put a Tensor on the server
CommandReply Redis::put_tensor(TensorBase& tensor)
{
    // Build the command
    Command cmd;
    cmd.add_field("AI.TENSORSET");
    cmd.add_field(tensor.name());
    cmd.add_field(tensor.type_str());
    cmd.add_fields(tensor.dims());
    cmd.add_field("BLOB");
    cmd.add_field_ptr(tensor.buf());

    // Run it
    return this->run(cmd);
}

// Get a Tensor from the server
CommandReply Redis::get_tensor(const std::string& key)
{
    // Build the command
    Command cmd;
    cmd.add_field("AI.TENSORGET");
    cmd.add_field(key);
    cmd.add_field("META");
    cmd.add_field("BLOB");

    // Run it
    return this->run(cmd);
}

// Rename a tensor in the database
CommandReply Redis::rename_tensor(const std::string& key,
                                  const std::string& new_key)
{
    // Build the command
    Command cmd;
    cmd.add_field("RENAME");
    cmd.add_field(key);
    cmd.add_field(new_key);

    // Run it
    return this->run(cmd);
}

// Delete a tensor in the database
CommandReply Redis::delete_tensor(const std::string& key)
{
    // Build the command
    Command cmd;
    cmd.add_field("DEL");
    cmd.add_field(key, true);

    // Run it
    return this->run(cmd);
}

// Copy a tensor from the source key to the destination key
CommandReply Redis::copy_tensor(const std::string& src_key,
                                const std::string& dest_key)
{
    //TODO can we do COPY for same hash slot or database?

    // Build a GET command to fetch the tensor
    Command cmd_get;
    cmd_get.add_field("AI.TENSORGET");
    cmd_get.add_field(src_key, true);
    cmd_get.add_field("META");
    cmd_get.add_field("BLOB");

    // Run the GET command
    CommandReply cmd_get_reply = this->run(cmd_get);
    if (cmd_get_reply.has_error() > 0) {
        throw std::runtime_error("Failed to retrieve tensor " +
                                 src_key + "from database");
    }

    // Decode the tensor
    std::vector<size_t> dims =
        CommandReplyParser::get_tensor_dims(cmd_get_reply);
    std::string_view blob =
        CommandReplyParser::get_tensor_data_blob(cmd_get_reply);
    TensorType type =
        CommandReplyParser::get_tensor_data_type(cmd_get_reply);

    // Build a PUT command to send the tensor back to the database
    // under the new key
    Command cmd_put;
    cmd_put.add_field("AI.TENSORSET");
    cmd_put.add_field(dest_key, true);
    cmd_put.add_field(TENSOR_STR_MAP.at(type));
    cmd_put.add_fields(dims);
    cmd_put.add_field("BLOB");
    cmd_put.add_field_ptr(blob);

    // Run the PUT command
    return this->run(cmd_put);
}

// Copy a vector of tensors from source keys to destination keys
CommandReply Redis::copy_tensors(const std::vector<std::string>& src,
                                 const std::vector<std::string>& dest)
{
    // Make sure vectors are the same length
    if (src.size() != dest.size()) {
        throw std::runtime_error("differing size vectors "\
                                 "passed to copy_tensors");
    }

    // Copy tensors one at a time. We only need to check one iterator
    // for reaching the end since we know from above that they are the
    // same length
    std::vector<std::string>::const_iterator it_src = src.cbegin();
    std::vector<std::string>::const_iterator it_dest = dest.cbegin();
    CommandReply reply;
    for ( ; it_src != src.cend(); it_src++, it_dest++) {
        reply = this->copy_tensor(*it_src, *it_dest);
        if (reply.has_error() > 0) {
            throw std::runtime_error("tensor copy failed");
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
    Command cmd;
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
    return this->run(cmd);
}

// Set a script from a string_view buffer in the database for future execution
CommandReply Redis::set_script(const std::string& key,
                               const std::string& device,
                               std::string_view script)
{
    // Build the command
    Command cmd;
    cmd.add_field("AI.SCRIPTSET");
    cmd.add_field(key, true);
    cmd.add_field(device);
    cmd.add_field("SOURCE");
    cmd.add_field_ptr(script);

    // Run it
    return this->run(cmd);
}

// Run a model in the database using the specificed input and output tensors
CommandReply Redis::run_model(const std::string& key,
                              std::vector<std::string> inputs,
                              std::vector<std::string> outputs)
{
    // Build the command
    Command cmd;
    cmd.add_field("AI.MODELRUN");
    cmd.add_field(key);
    cmd.add_field("INPUTS");
    cmd.add_fields(inputs);
    cmd.add_field("OUTPUTS");
    cmd.add_fields(outputs);

    // Run it
    return this->run(cmd);
}

// Run a script function in the database using the specificed input and
// output tensors
CommandReply Redis::run_script(const std::string& key,
                              const std::string& function,
                              std::vector<std::string> inputs,
                              std::vector<std::string> outputs)
{
    // Build the command
    Command cmd;
    cmd.add_field("AI.SCRIPTRUN");
    cmd.add_field(key);
    cmd.add_field(function);
    cmd.add_field("INPUTS");
    cmd.add_fields(inputs);
    cmd.add_field("OUTPUTS");
    cmd.add_fields(outputs);

    // Run it
    return this->run(cmd);
}

// Retrieve the model from the database
CommandReply Redis::get_model(const std::string& key)
{
    // Build the command
    Command cmd;
    cmd.add_field("AI.MODELGET");
    cmd.add_field(key);
    cmd.add_field("BLOB");

    // Run it
    return this->run(cmd);
}

// Retrieve the script from the database
CommandReply Redis::get_script(const std::string& key)
{
    // Build the command
    Command cmd;
    cmd.add_field("AI.SCRIPTGET");
    cmd.add_field(key, true);
    cmd.add_field("SOURCE");

    // Run it
    return this->run(cmd);
}

// Connect to the server at the address and port
inline void Redis::_connect(std::string address_port)
{
    // Note that this logic flow differs from a cluster
    // because the non-cluster Redis constructor
    // does not form a connection until a command is run

    this->_address_node_map.insert({address_port, nullptr});

    // Try to create the sw::redis::Redis object
    try {
        this->_redis = new sw::redis::Redis(address_port);
    }
    catch (std::exception& e) {
        this->_redis = NULL;
        throw std::runtime_error("Failed to create Redis object with error: " +
                                 std::string(e.what()));
    }
    catch (...) {
        this->_redis = NULL;
        throw std::runtime_error("A non-standard exception encountered "\
                                 "during client connection.");
    }

    // Attempt to have the sw::redis::Redis object
    // make a connection using the PING command
    int n_trials = 10;
    for (int i = 1; i <= n_trials; i++) {
        try {
            if (this->_redis->ping().compare("PONG") == 0) {
                return;
            }
            else if (i == n_trials) {
                throw std::runtime_error("Connection attempt failed");
            }
        }
        catch (std::exception& e) {
            if (i == n_trials) {
                throw std::runtime_error(e.what());
            }
        }
        catch (...) {
            if (i == n_trials) {
                throw std::runtime_error("A non-standard exception encountered"\
                                         " during client connection.");
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    // Should never get here
    throw std::runtime_error("End of _connect reached unexpectedly");
}

// EOF

