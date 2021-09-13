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

Redis::Redis() : RedisServer()
{
    std::string address_port = this->_get_ssdb();
    _add_to_address_map(address_port);
    this->_connect(address_port);
    return;
}

Redis::Redis(std::string address_port) : RedisServer()
{
    _add_to_address_map(address_port);
    this->_connect(address_port);
    return;
}

Redis::~Redis()
{
    if(this->_redis)
        delete this->_redis;
}

CommandReply Redis::run(SingleKeyCommand& cmd){
    return this->_run(cmd);
}

CommandReply Redis::run(MultiKeyCommand& cmd){
    return this->_run(cmd);
}

CommandReply Redis::run(CompoundCommand& cmd){
    return this->_run(cmd);
}

CommandReply Redis::run(AddressAtCommand& cmd){
    if (not is_addressable(cmd.get_address(), cmd.get_port()))
        throw std::runtime_error("The provided host and port do not match "\
                                 "the host and port used to initialize the "\
                                 "non-cluster client connection.");
    return this->_run(cmd);
}

CommandReply Redis::run(AddressAnyCommand& cmd){
    return this->_run(cmd);
}

std::vector<CommandReply> Redis::run(CommandList& cmds)
{
    std::vector<CommandReply> replies;
    CommandList::iterator cmd = cmds.begin();
    CommandList::iterator cmd_end = cmds.end();
    while(cmd != cmd_end) {
        replies.push_back(dynamic_cast<Command*>(*cmd)->run_me(this));
        cmd++;
    }
    return replies;
}

bool Redis::model_key_exists(const std::string& key)
{
    return this->key_exists(key);
}

bool Redis::key_exists(const std::string& key)
{
    SingleKeyCommand cmd;
    cmd.add_field("EXISTS");
    cmd.add_field(key);
    CommandReply reply = this->run(cmd);
    return reply.integer();
}

bool Redis::is_addressable(const std::string& address,
                           const uint64_t& port)
{
    return this->_address_node_map.find(address + ":"
                        + std::to_string(port))
                        != this->_address_node_map.end();
}

CommandReply Redis::put_tensor(TensorBase& tensor)
{
    SingleKeyCommand cmd;
    cmd.add_field("AI.TENSORSET");
    cmd.add_field(tensor.name());
    cmd.add_field(tensor.type_str());
    cmd.add_fields(tensor.dims());
    cmd.add_field("BLOB");
    cmd.add_field_ptr(tensor.buf());
    return this->run(cmd);
}

CommandReply Redis::get_tensor(const std::string& key)
{
    GetTensorCommand cmd;
    cmd.add_field("AI.TENSORGET");
    cmd.add_field(key);
    cmd.add_field("META");
    cmd.add_field("BLOB");
    return this->run(cmd);
}

CommandReply Redis::rename_tensor(const std::string& key,
                                  const std::string& new_key)
{
    MultiKeyCommand cmd;
    cmd.add_field("RENAME");
    cmd.add_field(key);
    cmd.add_field(new_key);
    return this->run(cmd);
}

CommandReply Redis::delete_tensor(const std::string& key)
{
    SingleKeyCommand cmd;
    cmd.add_field("DEL");
    cmd.add_field(key, true);
    return this->run(cmd);
}

CommandReply Redis::copy_tensor(const std::string& src_key,
                                const std::string& dest_key)
{
    //TODO can we do COPY for same hash slot or database?
    CommandReply cmd_get_reply;
    GetTensorCommand cmd_get;

    cmd_get.add_field("AI.TENSORGET");
    cmd_get.add_field(src_key, true);
    cmd_get.add_field("META");
    cmd_get.add_field("BLOB");
    cmd_get_reply = this->run(cmd_get);

    std::vector<size_t> dims =
        cmd_get.get_dims(cmd_get_reply);
    std::string_view blob =
        cmd_get.get_data_blob(cmd_get_reply);
    TensorType type =
        cmd_get.get_data_type(cmd_get_reply);

    CommandReply cmd_put_reply;
    MultiKeyCommand cmd_put;

    cmd_put.add_field("AI.TENSORSET");
    cmd_put.add_field(dest_key, true);
    cmd_put.add_field(TENSOR_STR_MAP.at(type));
    cmd_put.add_fields(dims);
    cmd_put.add_field("BLOB");
    cmd_put.add_field_ptr(blob);
    cmd_put_reply = this->run(cmd_put);

    return cmd_put_reply;
}

CommandReply Redis::copy_tensors(const std::vector<std::string>& src,
                                 const std::vector<std::string>& dest)
{
    std::vector<std::string>::const_iterator src_it = src.cbegin();
    std::vector<std::string>::const_iterator src_it_end = src.cend();

    std::vector<std::string>::const_iterator dest_it = dest.cbegin();
    std::vector<std::string>::const_iterator dest_it_end = dest.cend();

    CommandReply reply;

    while(src_it!=src_it_end && dest_it!=dest_it_end)
    {
        reply = this->copy_tensor(*src_it, *dest_it);
        src_it++;
        dest_it++;
    }
    return reply;
}


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
    SingleKeyCommand cmd;
    cmd.add_field("AI.MODELSET");
    cmd.add_field(model_name);
    cmd.add_field(backend);
    cmd.add_field(device);
    if(tag.size()>0) {
        cmd.add_field("TAG");
        cmd.add_field(tag);
    }
    if(batch_size>0) {
        cmd.add_field("BATCHSIZE");
        cmd.add_field(std::to_string(batch_size));
    }
    if(min_batch_size>0) {
        cmd.add_field("MINBATCHSIZE");
        cmd.add_field(std::to_string(min_batch_size));
    }
    if(inputs.size()>0) {
        cmd.add_field("INPUTS");
        cmd.add_fields(inputs);
    }
    if(outputs.size()>0) {
        cmd.add_field("OUTPUTS");
        cmd.add_fields(outputs);
    }
    cmd.add_field("BLOB");
    cmd.add_field_ptr(model);
    return this->run(cmd);
}

CommandReply Redis::set_script(const std::string& key,
                               const std::string& device,
                               std::string_view script)
{
    SingleKeyCommand cmd;
    cmd.add_field("AI.SCRIPTSET");
    cmd.add_field(key, true);
    cmd.add_field(device);
    cmd.add_field("SOURCE");
    cmd.add_field_ptr(script);
    return this->run(cmd);
}

CommandReply Redis::run_model(const std::string& key,
                              std::vector<std::string> inputs,
                              std::vector<std::string> outputs)
{
    CompoundCommand cmd;
    cmd.add_field("AI.MODELRUN");
    cmd.add_field(key);
    cmd.add_field("INPUTS");
    cmd.add_fields(inputs);
    cmd.add_field("OUTPUTS");
    cmd.add_fields(outputs);
    return this->run(cmd);
}

CommandReply Redis::run_script(const std::string& key,
                              const std::string& function,
                              std::vector<std::string> inputs,
                              std::vector<std::string> outputs)
{
    CompoundCommand cmd;
    cmd.add_field("AI.SCRIPTRUN");
    cmd.add_field(key);
    cmd.add_field(function);
    cmd.add_field("INPUTS");
    cmd.add_fields(inputs);
    cmd.add_field("OUTPUTS");
    cmd.add_fields(outputs);
    return this->run(cmd);
}

CommandReply Redis::get_model(const std::string& key)
{
    SingleKeyCommand cmd;
    cmd.add_field("AI.MODELGET");
    cmd.add_field(key);
    cmd.add_field("BLOB");
    return this->run(cmd);
}

CommandReply Redis::get_script(const std::string& key)
{
    SingleKeyCommand cmd;
    cmd.add_field("AI.SCRIPTGET");
    cmd.add_field(key, true);
    cmd.add_field("SOURCE");
    return this->run(cmd);
}

inline CommandReply Redis::_run(const Command& cmd)
{
    Command::const_iterator cmd_fields_start = cmd.cbegin();
    Command::const_iterator cmd_fields_end = cmd.cend();
    CommandReply reply;

    int n_trials = 100;

    while (n_trials > 0) {

        try {
            reply = this->_redis->command(cmd_fields_start, cmd_fields_end);

            if(reply.has_error()==0)
                return reply;
            n_trials = 0;
        }
        catch (sw::redis::IoError &e) {
            n_trials--;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        catch (sw::redis::ClosedError &e) {
            n_trials--;
            std::this_thread::sleep_for(std::chrono::seconds(2));
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
    }

    if (n_trials == 0) {
        if(reply.has_error()>0)
            reply.print_reply_error();
        throw std::runtime_error("Redis failed to execute command: " +
                                 cmd.first_field());
    }

    return reply;
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
        this->_redis = new sw::redis::Redis(address_port);
    }
    catch (std::exception& e) {
        this->_redis = NULL;
        throw std::runtime_error("Failed to create Redis "\
                                 "object with error: " +
                                 std::string(e.what()));
    }
    catch (...) {
        this->_redis = NULL;
        throw std::runtime_error("A non-standard exception "\
                                 "encountered during client "\
                                 "connection.");
    }

    // Attempt to have the sw::redis::Redis object
    // make a connection using the PING command
    int n_trials = 10;
    bool run_sleep = false;

    for(int i=1; i<=n_trials; i++) {
        run_sleep = false;
        try {
            if(this->_redis->ping().compare("PONG")==0) {
                break;
            }
            else {
                run_sleep = true;
            }
        }
        catch (std::exception& e) {
            if(i == n_trials) {
                throw std::runtime_error(e.what());
            }
            run_sleep = true;
        }
        catch (...) {
            if(i == n_trials) {
                throw std::runtime_error("A non-standard exception "\
                                         "encountered during client "\
                                         "connection.");
            }
            run_sleep = true;
        }
        if(run_sleep)
            std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    return;
}
