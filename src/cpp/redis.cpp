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
    this->_connect(address_port);
    return;
}

Redis::Redis(std::string address_port) : RedisServer()
{
    this->_connect(address_port);
    return;
}

Redis::~Redis()
{
    if(this->_redis)
        delete this->_redis;
}

CommandReply Redis::run(Command& cmd)
{
    Command::iterator cmd_fields_start = cmd.begin();
    Command::iterator cmd_fields_end = cmd.end();
    CommandReply reply;

    int n_trials = 100;
    bool success = true;

    while (n_trials > 0 && success) {

        try {
            reply = this->_redis->command(cmd_fields_start, cmd_fields_end);

            if(reply.has_error()==0)
                n_trials = -1;
            else
                n_trials = 0;
        }
        catch (sw::redis::TimeoutError &e) {
            n_trials--;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        catch (sw::redis::IoError &e) {
            n_trials--;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        catch (...) {
            n_trials--;
            throw;
        }
    }

    if (n_trials == 0)
        success = false;

    if (!success) {
        if(reply.has_error()>0)
            reply.print_reply_error();
        throw std::runtime_error("Redis failed to execute command: " +
                                 cmd.to_string());
    }

    return reply;
}

CommandReply Redis::run(CommandList& cmds)
{
    CommandList::iterator cmd = cmds.begin();
    CommandList::iterator cmd_end = cmds.end();
    CommandReply reply;
    while(cmd != cmd_end) {
        reply = this->run(**cmd);
        cmd++;
    }
    return reply;
}

bool Redis::model_key_exists(const std::string& key)
{
    return this->key_exists(key);
}

bool Redis::key_exists(const std::string& key)
{
    Command cmd;
    cmd.add_field("EXISTS");
    cmd.add_field(key);
    CommandReply reply = this->run(cmd);
    return reply.integer();
}

CommandReply Redis::put_tensor(TensorBase& tensor)
{
    Command cmd;
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
    Command cmd;
    cmd.add_field("AI.TENSORGET");
    cmd.add_field(key);
    cmd.add_field("META");
    cmd.add_field("BLOB");
    return this->run(cmd);
}

CommandReply Redis::rename_tensor(const std::string& key,
                                  const std::string& new_key)
{
    Command cmd;
    cmd.add_field("RENAME");
    cmd.add_field(key);
    cmd.add_field(new_key);
    return this->run(cmd);
}

CommandReply Redis::delete_tensor(const std::string& key)
{
    Command cmd;
    cmd.add_field("DEL");
    cmd.add_field(key, true);
    return this->run(cmd);
}

CommandReply Redis::copy_tensor(const std::string& src_key,
                                const std::string& dest_key)
{
    //TODO can we do COPY for same hash slot or database?
    CommandReply cmd_get_reply;
    Command cmd_get;

    cmd_get.add_field("AI.TENSORGET");
    cmd_get.add_field(src_key, true);
    cmd_get.add_field("META");
    cmd_get.add_field("BLOB");
    cmd_get_reply = this->run(cmd_get);

    std::vector<size_t> dims =
        CommandReplyParser::get_tensor_dims(cmd_get_reply);
    std::string_view blob =
        CommandReplyParser::get_tensor_data_blob(cmd_get_reply);
    TensorType type =
        CommandReplyParser::get_tensor_data_type(cmd_get_reply);

    CommandReply cmd_put_reply;
    Command cmd_put;

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
    Command cmd;
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
    Command cmd;
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
    Command cmd;
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
    Command cmd;
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
    Command cmd;
    cmd.add_field("AI.MODELGET");
    cmd.add_field(key);
    cmd.add_field("BLOB");
    return this->run(cmd);
}

CommandReply Redis::get_script(const std::string& key)
{
    Command cmd;
    cmd.add_field("AI.SCRIPTGET");
    cmd.add_field(key, true);
    cmd.add_field("SOURCE");
    return this->run(cmd);
}

inline void Redis::_connect(std::string address_port)
{
    int n_connection_trials = 10;

    while(n_connection_trials > 0) {
        try {
            this->_redis = new sw::redis::Redis(address_port);
            n_connection_trials = -1;
        }
        catch (sw::redis::TimeoutError &e) {
          std::cout << "WARNING: Caught redis TimeoutError: "
                    << e.what() << std::endl;
          std::cout << "WARNING: TimeoutError occurred with "\
                       "initial client connection.";
          std::cout << "WARNING: "<< n_connection_trials
                      << " more trials will be made.";
          n_connection_trials--;
          std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

    if(n_connection_trials==0)
        throw std::runtime_error("A connection could not be "\
                                 "established to the redis database.");
    return;
}
