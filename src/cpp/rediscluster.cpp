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

#include "rediscluster.h"
#include "nonkeyedcommand.h"
#include "keyedcommand.h"
#include "srexception.h"

using namespace SmartRedis;

// RedisCluster constructor
RedisCluster::RedisCluster() : RedisServer()
{
    std::string address_port = _get_ssdb();
    _connect(address_port);
    _map_cluster();
    if (_address_node_map.count(address_port) > 0)
        _last_prefix = _address_node_map.at(address_port)->prefix;
    else if (_db_nodes.size() > 0)
        _last_prefix = _db_nodes[0].prefix;
    else
        throw SRRuntimeException("Cluster mapping failed in client initialization");
}

// RedisCluster constructor. Uses address provided to constructor instead of
// environment variables
RedisCluster::RedisCluster(std::string address_port) : RedisServer()
{
    _connect(address_port);
    _map_cluster();
    if (_address_node_map.count(address_port) > 0)
        _last_prefix = _address_node_map.at(address_port)->prefix;
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

// Run a compound Command on the server
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
    if (is_addressable(cmd.get_address(), cmd.get_port()))
        db_prefix = _address_node_map.at(cmd.get_address() + ":"
                    + std::to_string(cmd.get_port()))->prefix;
    else
        throw SRRuntimeException("Redis has failed to find database");

    return _run(cmd, db_prefix);
}

// Run a non-keyed Command that addresses any db node on the server
CommandReply RedisCluster::run(AddressAnyCommand &cmd)
{
    return _run(cmd, _last_prefix);
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
    cmd.add_field("EXISTS");
    cmd.add_field(key, true);

    // Run it
    CommandReply reply = run(cmd);
    if (reply.has_error() > 0)
        throw SRRuntimeException("Error encountered while checking "\
                                  "for existence of key " + key);
    return (bool)reply.integer();
}

// Check if a key exists in the database
bool RedisCluster::is_addressable(const std::string& address,
                                  const uint64_t& port)
{
    std::string addr = address + ":" + std::to_string(port);
    return _address_node_map.find(addr) != _address_node_map.end();
}

// Put a Tensor on the server
CommandReply RedisCluster::put_tensor(TensorBase& tensor)
{
    // Build the command
    SingleKeyCommand cmd;
    cmd.add_field("AI.TENSORSET");
    cmd.add_field(tensor.name(), true);
    cmd.add_field(tensor.type_str());
    cmd.add_fields(tensor.dims());
    cmd.add_field("BLOB");
    cmd.add_field_ptr(tensor.buf());

    // Run it
    return run(cmd);
}

// Get a Tensor from the server
CommandReply RedisCluster::get_tensor(const std::string& key)
{
    // Build the command
    GetTensorCommand cmd;
    cmd.add_field("AI.TENSORGET");
    cmd.add_field(key, true);
    cmd.add_field("META");
    cmd.add_field("BLOB");

    // Run it
    return run(cmd);
}

// Rename a tensor in the database
CommandReply RedisCluster::rename_tensor(const std::string& key,
                                         const std::string& new_key)
{
    // Check wehether we have to switch hash slots
    uint16_t key_hash_slot = _get_hash_slot(key);
    uint16_t new_key_hash_slot = _get_hash_slot(new_key);

    // If not, we can use a simple RENAME command
    CommandReply reply;
    if (key_hash_slot == new_key_hash_slot) {
        // Build the command
        CompoundCommand cmd;
        cmd.add_field("RENAME");
        cmd.add_field(key, true);
        cmd.add_field(new_key, true);

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
    cmd.add_field("UNLINK");
    cmd.add_field(key, true);

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
    cmd_get.add_field("AI.TENSORGET");
    cmd_get.add_field(src_key, true);
    cmd_get.add_field("META");
    cmd_get.add_field("BLOB");

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
                                     std::string_view model,
                                     const std::string& backend,
                                     const std::string& device,
                                     int batch_size,
                                     int min_batch_size,
                                     const std::string& tag,
                                     const std::vector<std::string>& inputs,
                                     const std::vector<std::string>& outputs)
{
    std::vector<DBNode>::const_iterator node = _db_nodes.cbegin();
    CommandReply reply;
    for ( ; node != _db_nodes.cend(); node++) {
        // Build the node prefix
        std::string prefixed_key = "{" + node->prefix + "}." + model_name;

        // Build the MODELSET commnd
        CompoundCommand cmd;
        cmd.add_field("AI.MODELSET");
        cmd.add_field(prefixed_key, true);
        cmd.add_field(backend);
        cmd.add_field(device);

        // Add optional fields as requested
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
        if ( inputs.size() > 0) {
            cmd.add_field("INPUTS");
            cmd.add_fields(inputs);
        }
        if (outputs.size() > 0) {
            cmd.add_field("OUTPUTS");
            cmd.add_fields(outputs);
        }
        cmd.add_field("BLOB");
        cmd.add_field_ptr(model);

        // Run the command
        reply = run(cmd);
        if (reply.has_error() > 0) {
            throw SRRuntimeException("SetModel failed for node " + node->name);
        }
    }

    // Done
    return reply;
}

// Set a script from a string buffer in the database for future execution
CommandReply RedisCluster::set_script(const std::string& key,
                                      const std::string& device,
                                      std::string_view script)
{
    CommandReply reply;
    std::vector<DBNode>::const_iterator node = _db_nodes.cbegin();
    for ( ; node != _db_nodes.cend(); node++) {
        // Build the node prefix
        std::string prefix_key = "{" + node->prefix + "}." + key;

        // Build the SCRIPTSET command
        SingleKeyCommand cmd;
        cmd.add_field("AI.SCRIPTSET");
        cmd.add_field(prefix_key, true);
        cmd.add_field(device);
        cmd.add_field("SOURCE");
        cmd.add_field_ptr(script);

        // Run the command
        reply = run(cmd);
        if (reply.has_error() > 0) {
            throw SRRuntimeException("SetModel failed for node " + node->name);
        }
    }

    // Done
    return reply;
}

// Run a model in the database using the specificed input and output tensors
CommandReply RedisCluster::run_model(const std::string& key,
                                     std::vector<std::string> inputs,
                                     std::vector<std::string> outputs)
{
    /*  For this version of run model, we have to copy all
        input and output tensors, so we will randomly select
        a model.  We can't use rand, because MPI would then
        have the same random number across all ranks.  Instead
        We will choose it based on the db of the first input tensor.
    */

    uint16_t hash_slot = _get_hash_slot(inputs[0]);
    uint16_t db_index = _get_dbnode_index(hash_slot, 0,
                                                _db_nodes.size()-1);
    DBNode* db = &(_db_nodes[db_index]);
    if (db == NULL) {
        throw SRRuntimeException("Missing DB node found in run_model");
    }

    // Generate temporary names so that all keys go to same slot
    std::vector<std::string> tmp_inputs = _get_tmp_names(inputs, db->prefix);
    std::vector<std::string> tmp_outputs = _get_tmp_names(outputs, db->prefix);

    // Copy all input tensors to temporary names to align hash slots
    copy_tensors(inputs, tmp_inputs);

    // Build the MODELRUN command
    std::string model_name = "{" + db->prefix + "}." + std::string(key);
    CompoundCommand cmd;
    cmd.add_field("AI.MODELRUN");
    cmd.add_field(model_name, true);
    cmd.add_field("INPUTS");
    cmd.add_fields(tmp_inputs);
    cmd.add_field("OUTPUTS");
    cmd.add_fields(tmp_outputs);

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

// Run a script function in the database using the specificed input
// and output tensors
CommandReply RedisCluster::run_script(const std::string& key,
                                      const std::string& function,
                                      std::vector<std::string> inputs,
                                      std::vector<std::string> outputs)
{
    // Locate the DB node for the script
    uint16_t hash_slot = _get_hash_slot(inputs[0]);
    uint16_t db_index = _get_dbnode_index(hash_slot, 0, _db_nodes.size() - 1);
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
    CommandReply reply;
    cmd.add_field("AI.SCRIPTRUN");
    cmd.add_field(script_name, true);
    cmd.add_field(function);
    cmd.add_field("INPUTS");
    cmd.add_fields(tmp_inputs);
    cmd.add_field("OUTPUTS");
    cmd.add_fields(tmp_outputs);

    // Run it
    reply = run(cmd);
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

// Retrieve the model from the database
CommandReply RedisCluster::get_model(const std::string& key)
{
    // Build the node prefix
    std::string prefixed_str = "{" + _db_nodes[0].prefix + "}." + key;

    // Build the MODELGET command
    SingleKeyCommand cmd;
    cmd.add_field("AI.MODELGET");
    cmd.add_field(prefixed_str, true);
    cmd.add_field("BLOB");

    // Run it
    return run(cmd);
}

// Retrieve the script from the database
CommandReply RedisCluster::get_script(const std::string& key)
{
    std::string prefixed_str = "{" + _db_nodes[0].prefix + "}." + key;

    SingleKeyCommand cmd;
    cmd.add_field("AI.SCRIPTGET");
    cmd.add_field(prefixed_str, true);
    cmd.add_field("SOURCE");
    return run(cmd);
}

inline CommandReply RedisCluster::_run(const Command& cmd, std::string db_prefix)
{
    const int n_trials = 100;
    std::string_view sv_prefix(db_prefix.data(), db_prefix.size());

    // Execute the commmand
    CommandReply reply;
    bool executed = false;
    for (int trial = 0; trial < n_trials; trial++) {
        try {
            sw::redis::Redis db = _redis_cluster->redis(sv_prefix, false);
            reply = db.command(cmd.cbegin(), cmd.cend());
            if (reply.has_error() == 0) {
                _last_prefix = db_prefix;
                return reply;
            }
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
            throw SRInternalException("A non-standard exception encountered "\
                                       "during command " + cmd.first_field() +
                                       " execution.");
        }

        // Sleep before the next attempt
        std::this_thread::sleep_for(std::chrono::seconds(2));
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

// Connect to the cluster at the address and port
inline void RedisCluster::_connect(std::string address_port)
{
    const int n_trials = 10;
    for (int i = 1; i <= n_trials; i++) {
        try {
            _redis_cluster = new sw::redis::RedisCluster(address_port);
            return;
        }
        catch (sw::redis::Error& e) {
            _redis_cluster = NULL;
            throw SRDatabaseException(std::string("Unable to connect to "\
                                    "backend Redis database: ") +
                                    e.what());
        }
        catch (std::bad_alloc& e) {
            throw SRBadAllocException("RedisCluster connection");
        }
        catch (std::exception& e) {
            if (_redis_cluster != NULL) {
                delete _redis_cluster;
                _redis_cluster = NULL;
            }
            if (i == n_trials) {
                throw SRRuntimeException(e.what());
            }
        }
        catch (...) {
            if (_redis_cluster != NULL) {
                delete _redis_cluster;
                _redis_cluster = NULL;
            }
            if (i == n_trials) {
                throw SRInternalException("A non-standard exception was "\
                                           "encountered during client "\
                                           "connection.");
            }
        }

        // Sleep before the next attempt
        if (_redis_cluster != NULL) {
            delete _redis_cluster;
            _redis_cluster = NULL;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    // If we get here, we failed to establish a connection
    throw SRTimeoutException(std::string("Connection attempt failed after ") +
                                         std::to_string(n_trials) + "tries");
}

// Map the RedisCluster via the CLUSTER SLOTS command
inline void RedisCluster::_map_cluster()
{
    // Clear out our old map
    _db_nodes.clear();
    _address_node_map.clear();

    // Build the CLUSTER SLOTS command
    AddressAnyCommand cmd;
    cmd.add_field("CLUSTER");
    cmd.add_field("SLOTS");

    // Run it
    CommandReply reply(_redis_cluster->
                 command(cmd.begin(), cmd.end()));
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
        uint16_t hash_slot = _get_hash_slot(*key_it);
        uint16_t db_index = _get_dbnode_index(hash_slot, 0,
                                           _db_nodes.size() - 1);
        if (prefix.size() == 0) {
            prefix = _db_nodes[db_index].prefix;
        }
        else if (prefix != _db_nodes[db_index].prefix) {
            throw SRRuntimeException("Multi-key commands are "\
                                      "not valid: " +
                                      cmd.to_string());
        }
    }

    // Done
    return prefix;
}

// Process the CommandReply for CLUSTER SLOTS to build DBNode information
inline void RedisCluster::_parse_reply_for_slots(CommandReply& reply)
{
    /* Each reply element of the main message, of which there should
    be n_db_nodes, is:
    0) (integer) min slot
    1) (integer) max slot
    2) 0) "ip address"
       1) (integer) port
       2) "name"
    */
    size_t n_db_nodes = reply.n_elements();
    _db_nodes = std::vector<DBNode>(n_db_nodes);

    for (size_t i = 0; i < n_db_nodes; i++) {
        _db_nodes[i].lower_hash_slot = reply[i][0].integer();
        _db_nodes[i].upper_hash_slot = reply[i][1].integer();
        _db_nodes[i].ip = std::string(reply[i][2][0].str(),
                                            reply[i][2][0].str_len());
        _db_nodes[i].port = reply[i][2][1].integer();
        _db_nodes[i].name = std::string(reply[i][2][2].str(),
                                              reply[i][2][2].str_len());
        _db_nodes[i].prefix = _get_crc16_prefix(_db_nodes[i].lower_hash_slot);
        _address_node_map.insert({_db_nodes[i].ip + ":"
                                    + std::to_string(_db_nodes[i].port),
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
uint16_t RedisCluster::_get_dbnode_index(uint16_t hash_slot,
                                   unsigned lhs, unsigned rhs)
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
            return _get_dbnode_index(hash_slot, lhs, m - 1);
        else
            return _get_dbnode_index(hash_slot, m + 1, rhs);
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
    cmd.add_field("DEL");
    cmd.add_fields(keys, true);

    // Run it, ignoring failure
    (void)run(cmd);
}

// Run a model in the database that uses dagrun
void RedisCluster::__run_model_dagrun(const std::string& key,
                                      std::vector<std::string> inputs,
                                      std::vector<std::string> outputs)
{
    /* This function will run a RedisAI model.  Because the RedisAI
    AI.RUNMODEL and AI.DAGRUN commands assume that the tensors
    and model are all on the same node.  As a result, we will
    have to retrieve all input tensors that are not on the same
    node as the model and set temporary
    */

    // TODO We need to make sure that no other clients are using the
    // same keys and model because we may end up overwriting or having
    // race conditions on who can use the model, etc.

    DBNode* db = _get_model_script_db(key, inputs, outputs);

    // Create list of input tensors that do not hash to db slots
    std::unordered_set<std::string> remote_inputs;
    for (size_t i = 0; i < inputs.size(); i++) {
        uint16_t hash_slot = _get_hash_slot(inputs[i]);
        if (hash_slot < db->lower_hash_slot ||
            hash_slot > db->upper_hash_slot) {
            remote_inputs.insert(inputs[i]);
        }
    }

    // Retrieve tensors that do not hash to db,
    // rename the tensors to {prefix}.tensor_name.TMP
    // TODO we need to make sure users don't use the .TMP suffix
    // or check that the key does not exist
    for (size_t i = 0; i < inputs.size(); i++) {
        if (remote_inputs.count(inputs[i]) > 0) {
            std::string new_key = "{" + db->prefix + "}." + inputs[i] + ".TMP";
            copy_tensor(inputs[i], new_key);
            remote_inputs.erase(inputs[i]);
            remote_inputs.insert(new_key);
            inputs[i] = new_key;
        }
    }

    // Create a renaming scheme for output tensor
    std::unordered_map<std::string, std::string> remote_outputs;
    for (size_t i = 0; i < outputs.size(); i++) {
        uint16_t hash_slot = _get_hash_slot(outputs[i]);
        if (hash_slot < db->lower_hash_slot ||
            hash_slot > db->upper_hash_slot) {
            std::string tmp = "{" + db->prefix + "}." + outputs[i] + ".TMP";
            remote_outputs.insert({outputs[i], tmp});
            outputs[i] = remote_outputs[outputs[i]];
        }
    }

    // Build the DAGRUN command
    std::string model_name = "{" + db->prefix + "}." + key;
    CompoundCommand cmd;
    cmd.add_field("AI.DAGRUN");
    cmd.add_field("LOAD");
    cmd.add_field(std::to_string(inputs.size()));
    cmd.add_fields(inputs);
    cmd.add_field("PERSIST");
    cmd.add_field(std::to_string(outputs.size()));
    cmd.add_fields(outputs);
    cmd.add_field("|>");
    cmd.add_field("AI.MODELRUN");
    cmd.add_field(model_name, true);
    cmd.add_field("INPUTS");
    cmd.add_fields(inputs);
    cmd.add_field("OUTPUTS");
    cmd.add_fields(outputs);

    // Run it
    CommandReply reply = run(cmd);
    if (reply.has_error()) {
        throw SRRuntimeException("Failed to execute DAGRUN");
    }

    // Delete temporary input tensors
    std::unordered_set<std::string>::const_iterator i_it =
        remote_inputs.begin();
    for ( ; i_it !=  remote_inputs.end(); i_it++)
        delete_tensor(*i_it);

    // Move temporary output to the correct location and
    // delete temporary output tensors
    std::unordered_map<std::string, std::string>::const_iterator j_it =
        remote_outputs.begin();
    for ( ; j_it != remote_outputs.end(); j_it++)
        rename_tensor(j_it->second, j_it->first);
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
        uint16_t db_index = _get_dbnode_index(hash_slot, 0, _db_nodes.size());
        hash_slot_tally[db_index]++;
    }

    for (size_t i = 0; i < outputs.size(); i++) {
        uint16_t hash_slot = _get_hash_slot(outputs[i]);
        uint16_t db_index = _get_dbnode_index(hash_slot, 0, _db_nodes.size());
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
