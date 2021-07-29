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

#ifndef SMARTREDIS_CPP_COMMANDREPLYPARSER_H
#define SMARTREDIS_CPP_COMMANDREPLYPARSER_H

#include <vector>
#include "commandreply.h"
#include "tensorbase.h"
#include "enums/cpp_tensor_type.h"

using parsed_reply_map = std::unordered_map<std::string, std::string>;
using parsed_reply_nested_map = std::unordered_map<std::string, parsed_reply_map>;

namespace SmartRedis {

namespace CommandReplyParser {

inline std::vector<size_t> get_tensor_dims(CommandReply& reply)
{
    /* This function will fill a vector with the dimensions of the
    tensor.  We assume right now that the META reply is always
    in the same order and we can index base reply elements array;
    */

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

inline std::string_view get_tensor_data_blob(CommandReply& reply)
{
    /* Returns a string view of the data tensor blob value
    */

    //We are going to assume right now that the meta data reply
    //is always in the same order and we are going to just
    //index into the base reply.

    if(reply.n_elements() < 6)
        throw std::runtime_error("The message does not have the "\
                                "correct number of fields");

    return std::string_view(reply[5].str(), reply[5].str_len());
}

inline TensorType get_tensor_data_type(CommandReply& reply)
{
    /* Returns a string of the tensor type
    */

    //We are going to assume right now that the meta data reply
    //is always in the same order and we are going to just
    //index into the base reply.

    if(reply.n_elements() < 2)
        throw std::runtime_error("The message does not have the correct "\
                                "number of fields");

    return TENSOR_TYPE_MAP.at(std::string(reply[1].str(),
                                            reply[1].str_len()));
}

inline parsed_reply_nested_map parse_db_node_info(std::string info)
{
    /* Parse database node information from get_db_node_info()
    *          into a nested unordered_map
    *  Returns parsed_reply_nested_map containing the database node information
    */
    parsed_reply_nested_map info_map;

    std::string delim = "\r\n";
    std::string currKey = "";
    size_t start = 0U;
    size_t end = info.find(delim);

    while (end != std::string::npos)
    {
        std::string line = info.substr(start, end-start);
        start = end + delim.length();
        end = info.find(delim, start);
        if (line.length() == 0)
            continue;
        if (line[0] == '#')
        {
            currKey = line.substr(2);
            if (info_map.find(currKey)==info_map.end())
                info_map[currKey] = {};
        }
        else
        {
            std::size_t separatorIdx = line.find(':');
            info_map[currKey][line.substr(0, separatorIdx)] =
                                line.substr(separatorIdx+1);
        }
    }
    std::string line = info.substr(start);
    if (line.length() > 0)
    {
        std::size_t separatorIdx = line.find(':');
        if (info_map.find(currKey)==info_map.end())
                    info_map[currKey] = {};
        info_map[currKey][line.substr(0, separatorIdx)] =
                            line.substr(separatorIdx+1);
    }
    return info_map;
}

inline parsed_reply_map parse_db_cluster_info(std::string info)
{
    /* Parse database node information from get_db_node_info()
    *          into a nested unordered_map
    *  Returns parsed_reply_map containing the database node cluster
    *  information
    */
    parsed_reply_map info_map;

    std::string delim = "\r\n";
    std::string currKey = "";
    size_t start = 0U;
    size_t end = info.find(delim);

    while (end != std::string::npos)
    {
        std::string line = info.substr(start, end-start);
        start = end + delim.length();
        end = info.find(delim, start);
        if (line.length() == 0)
            continue;

        std::size_t separatorIdx = line.find(':');
        info_map[line.substr(0, separatorIdx)] =
                 line.substr(separatorIdx+1);
    }
    std::string line = info.substr(start);
    if (line.length() > 0)
    {
        std::size_t separatorIdx = line.find(':');
        info_map[line.substr(0, separatorIdx)] =
                 line.substr(separatorIdx+1);
    }
    return info_map;
}

} //namespace CommandReplyParser

} //namespace SmartRedis

#endif //SMARTREDIS_CPP_COMMANDREPLYPARSER_H
