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

#ifndef SMARTREDIS_UTILITY_H
#define SMARTREDIS_UTILITY_H

#include <cstdlib>
#include <string>
#include "sr_enums.h"

///@file

namespace SmartRedis {

/*!
*   \brief  Flag to skip warnings when retrieving configuration options
*           and the requested option is not present
*/
const int flag_suppress_warning = 1;

/*!
*   \brief  Flag to emit a KeyException when retrieving configuration options
*           and the requested option is not present
*/
const int throw_on_absent = 2;

/*!
*   \brief Initialize an integer from configuration, such as an
*          environment variable
*   \param value Receives the configuration value
*   \param cfg_key The key to query for the configuration variable
*   \param default_value Default if configuration key is not set
*   \param flags flag_suppress_warning = Do not issue a warning if the
*                variable is not set; throw_on_absent = throw KeyException
*                if value not set. The value zero means that no flags are set
*   \throw KeyException if value not set and throw_on_absent is not set
*/
void get_config_integer(int& value,
                        const std::string& cfg_key,
                        const int default_value,
                        int flags = 0);

/*!
*   \brief Initialize an string from configuration, such as an
*          environment variable
*   \param value Receives the configuration value
*   \param cfg_key The key to query for the configuration variable
*   \param default_value Default if configuration key is not set
*   \param flags flag_suppress_warning = Do not issue a warning if the
*                variable is not set; throw_on_absent = throw KeyException
*                if value not set. The value zero means that no flags are set
*   \throw KeyException if value not set and throw_on_absent is not set
*/
void get_config_string(std::string& value,
                       const std::string& cfg_key,
                       const std::string& default_value,
                       int flags = 0);

/*!
*   \brief Create a string representation of a tensor type
*   \param ttype The tensor type to put in string form
*/
std::string to_string(SRTensorType ttype);

/*!
*   \brief Create a string representation of a metadata field type
*   \param mdtype The metadata field type to put in string form
*/
std::string to_string(SRMetaDataType mdtype);


} // namespace SmartRedis

#endif // SMARTREDIS_UTILITY_H
