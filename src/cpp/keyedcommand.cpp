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

#include "keyedcommand.h"
#include "redisserver.h"

using namespace SmartRedis;

CommandReply KeyedCommand::runme(RedisServer * r)
{
    CommandReply reply;
    return reply;
}

CommandReply MultiKeyCommand::runme(RedisServer * r)
{
    return r->run(*this);
}

CommandReply SingleKeyCommand::runme(RedisServer * r)
{
    return r->run(*this);
}

CommandReply CompoundCommand::runme(RedisServer * r)
{
    return r->run(*this);
}

CommandReply GetTensorCommand::runme(RedisServer * r)
{
    return r->run(*this);
}

std::vector<size_t> GetTensorCommand::get_tensor_dims(CommandReply& reply)
{
    if(reply.n_elements() < 6)
        throw std::runtime_error("The message does not have the "\
                                "correct number of fields");

    size_t n_dims = reply[3].n_elements();
    std::vector<size_t> dims(n_dims);

    for(size_t i=0; i<n_dims; i++) {
        dims[i] = reply[3][i].integer();
    }

    return dims;
}

TensorType GetTensorCommand::get_tensor_data_type(CommandReply& reply)
{
    if(reply.n_elements() < 2)
        throw std::runtime_error("The message does not have the correct "\
                                "number of fields");

    return TENSOR_TYPE_MAP.at(std::string(reply[1].str(),
                                            reply[1].str_len()));
}

std::string_view GetTensorCommand::get_tensor_data_blob(CommandReply& reply)
{
    if(reply.n_elements() < 6)
        throw std::runtime_error("The message does not have the "\
                                "correct number of fields");

    return std::string_view(reply[5].str(), reply[5].str_len());
}