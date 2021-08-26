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

#include "dbinfocommand.h"
#include "redisserver.h"

using namespace SmartRedis;

// Parse database info into nested unordered map
parsed_reply_nested_map DBInfoCommand::parse_db_node_info(std::string info)
{
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

    return info_map;
}