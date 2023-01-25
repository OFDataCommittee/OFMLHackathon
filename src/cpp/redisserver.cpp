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

#include <ctype.h>
#include "redisserver.h"
#include "srexception.h"
#include "utility.h"
#include "srobject.h"

using namespace SmartRedis;

// RedisServer constructor
RedisServer::RedisServer(const SRObject* context)
    : _context(context), _gen(_rd())
{
    get_config_integer(_connection_timeout, _CONN_TIMEOUT_ENV_VAR,
                         _DEFAULT_CONN_TIMEOUT);
    get_config_integer(_connection_interval, _CONN_INTERVAL_ENV_VAR,
                         _DEFAULT_CONN_INTERVAL);
    get_config_integer(_command_timeout, _CMD_TIMEOUT_ENV_VAR,
                         _DEFAULT_CMD_TIMEOUT);
    get_config_integer(_command_interval, _CMD_INTERVAL_ENV_VAR,
                         _DEFAULT_CMD_INTERVAL);
    get_config_integer(_thread_count, _TP_THREAD_COUNT,
                         _DEFAULT_THREAD_COUNT);

    _check_runtime_variables();

    _connection_attempts = (_connection_timeout * 1000) /
                            _connection_interval + 1;

    _command_attempts = (_command_timeout * 1000) /
                         _command_interval + 1;

    _tp = new ThreadPool(_context, _thread_count);
}

// RedisServer destructor
RedisServer::~RedisServer()
{
    // Terminate the thread pool
    _tp->shutdown();
    delete _tp;
}


// Retrieve a single address, randomly chosen from a list of addresses if
// applicable, from the SSDB environment variable
SRAddress RedisServer::_get_ssdb()
{
    // Retrieve the environment variable
    std::string db_spec;
    get_config_string(db_spec, "SSDB", "");
    if (db_spec.length() == 0)
        throw SRRuntimeException("The environment variable SSDB "\
                                 "must be set to use the client.");
    _check_ssdb_string(db_spec);

    // Parse the data in it
    std::vector<SRAddress> address_choices;
    const char delim = ',';

    size_t i_pos = 0;
    size_t j_pos = db_spec.find(delim);
    while (j_pos != std::string::npos) {
        std::string substr = db_spec.substr(i_pos, j_pos - i_pos);
        SRAddress addr_spec(substr);
        address_choices.push_back(addr_spec);
        i_pos = j_pos + 1;
        j_pos = db_spec.find(delim, i_pos);
    }
    // Catch the last value that does not have a trailing ','
    if (i_pos < db_spec.size()) {
        std::string substr = db_spec.substr(i_pos, j_pos - i_pos);
        SRAddress addr_spec(substr);
        address_choices.push_back(addr_spec);
    }

    // Pick an entry from the list at random
    std::uniform_int_distribution<> distrib(0, address_choices.size() - 1);
    return address_choices[distrib(_gen)];
}

// Check that the SSDB environment variable value does not have any errors
void RedisServer::_check_ssdb_string(const std::string& env_str) {
    for (size_t i = 0; i < env_str.size(); i++) {
        char c = env_str[i];
        if (!isalnum(c) && c != '.' && c != ':' && c != ',' && c != '/') {
            throw SRRuntimeException("The provided SSDB value, " + env_str +
                                     " is invalid because of character " + c);
        }
    }
}

// Check that runtime variables are within valid ranges
inline void RedisServer::_check_runtime_variables()
{
    if (_connection_timeout <= 0) {
        throw SRParameterException(_CONN_TIMEOUT_ENV_VAR +
                                   " must be greater than 0.");
    }

    if (_connection_interval <= 0) {
        throw SRParameterException(_CONN_INTERVAL_ENV_VAR +
                                   " must be greater than 0.");
    }

    if (_command_timeout <= 0) {
        throw SRParameterException(_CMD_TIMEOUT_ENV_VAR + " " +
                                   std::to_string(_command_timeout) +
                                   " must be greater than 0.");
    }

    if (_command_interval <= 0) {
        throw SRParameterException(_CMD_INTERVAL_ENV_VAR +
                                   " must be greater than 0.");
    }

    if (_connection_timeout > (INT_MAX / 1000)) {
        throw SRParameterException(_CONN_TIMEOUT_ENV_VAR +
                                   " must be less than "
                                   + std::to_string(INT_MAX / 1000));
    }

    if (_command_timeout > (INT_MAX / 1000)) {
        throw SRParameterException(_CMD_TIMEOUT_ENV_VAR + " " +
                                   std::to_string(_command_timeout) +
                                   " must be less than "
                                   + std::to_string(INT_MAX / 1000));
    }
}