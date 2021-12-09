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

#include "gettensorcommand.h"
#include "redisserver.h"
#include "srexception.h"

using namespace SmartRedis;

const size_t num_elements = 2;
const size_t num_fields = 6;
const int type_idx = 1;
const int dims_idx = 3;
const int blob_idx = 5;

// Run GetTensorCommand on the server
CommandReply GetTensorCommand::run_me(RedisServer* server)
{
    return server->run(*this);
}

// Returns a filled vector with the dimensions of the tensor
std::vector<size_t> GetTensorCommand::get_dims(CommandReply& reply)
{
    if (reply.n_elements() < num_fields) {
        throw SRRuntimeException("The message does not have the "\
                                  "correct number of fields");
    }

    size_t n_dims = reply[dims_idx].n_elements();
    std::vector<size_t> dims(n_dims);

    for (size_t i = 0; i < n_dims; i++) {
        dims[i] = reply[dims_idx][i].integer();
    }

    return dims;
}

// Returns a string of the tensor type
SRTensorType GetTensorCommand::get_data_type(CommandReply& reply)
{
    if (reply.n_elements() < num_elements) {
        throw SRRuntimeException("The message does not have the correct "\
                                  "number of fields");
    }

    return TENSOR_TYPE_MAP.at(std::string(reply[type_idx].str(),
                                          reply[type_idx].str_len()));
}

// Returns a string view of the data tensor blob value
std::string_view GetTensorCommand::get_data_blob(CommandReply& reply)
{
    if (reply.n_elements() < num_fields) {
        throw SRRuntimeException("The message does not have the "\
                                  "correct number of fields");
    }

    return std::string_view(reply[blob_idx].str(), reply[blob_idx].str_len());
}
