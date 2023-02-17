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

#ifndef SMARTREDIS_DBINFOCOMMAND_H
#define SMARTREDIS_DBINFOCOMMAND_H

#include "addressatcommand.h"

///@file

/*!
*  \brief A nested map of reply data from a database response
*/
using parsed_reply_nested_map = std::unordered_map<std::string,
                                std::unordered_map<std::string, std::string>>;

namespace SmartRedis {

class RedisServer;

/*!
*   \brief The DBInfoCommand class constructs the Redis DB INFO command.
*/
class DBInfoCommand : public AddressAtCommand
{
    public:
        /*!
        *   \brief Parse database node information from get_db_node_info()
        *          into a nested unordered_map
        *   \param info containing the database node information
        *   \return parsed_reply_nested_map containing the database node information
        */
        static parsed_reply_nested_map parse_db_node_info(std::string info);
};

} // namespace SmartRedis

#endif // SMARTREDIS_DBINFOCOMMAND_H
