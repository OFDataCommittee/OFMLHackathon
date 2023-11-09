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
#include <sw/redis++/redis++.h>
#include "redisserver.h"
#include "srexception.h"
#include "utility.h"
#include "srobject.h"
#include "configoptions.h"

using namespace SmartRedis;

// RedisServer constructor
RedisServer::RedisServer(ConfigOptions* cfgopts)
    : _cfgopts(cfgopts), _context(cfgopts->_get_log_context()),
      _gen(_rd())
{
    _connection_timeout = _cfgopts->_resolve_integer_option(
        _CONN_TIMEOUT_ENV_VAR, _DEFAULT_CONN_TIMEOUT);
    _connection_interval = _cfgopts->_resolve_integer_option(
        _CONN_INTERVAL_ENV_VAR, _DEFAULT_CONN_INTERVAL);
    _command_timeout = _cfgopts->_resolve_integer_option(
        _CMD_TIMEOUT_ENV_VAR, _DEFAULT_CMD_TIMEOUT);
    _command_interval = _cfgopts->_resolve_integer_option(
        _CMD_INTERVAL_ENV_VAR, _DEFAULT_CMD_INTERVAL);
    _thread_count = _cfgopts->_resolve_integer_option(
        _TP_THREAD_COUNT, _DEFAULT_THREAD_COUNT);

    _check_runtime_variables();

    _connection_attempts = (_connection_timeout * 1000) /
                            _connection_interval + 1;

    _command_attempts = (_command_timeout * 1000) /
                         _command_interval + 1;

    _tp = new ThreadPool(_context, _thread_count);
    _model_chunk_size = _UNKNOWN_MODEL_CHUNK_SIZE;
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
    std::string db_spec = _cfgopts->_resolve_string_option("SSDB", "");
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

    std::string msg = "Found " + std::to_string(address_choices.size()) + " addresses:";
    _cfgopts->_get_log_context()->log_data(LLDeveloper, msg);
    for (size_t i = 0; i < address_choices.size(); i++) {
        _cfgopts->_get_log_context()->log_data(
            LLDeveloper, "\t" + address_choices[i].to_string());
    }

    // Pick an entry from the list at random
    std::uniform_int_distribution<> distrib(0, address_choices.size() - 1);
    auto choice = address_choices[distrib(_gen)];
    _cfgopts->_get_log_context()->log_data(
        LLDeveloper, "Picked: " + choice.to_string());
    return choice;
}

// Check that the SSDB environment variable value does not have any errors
void RedisServer::_check_ssdb_string(const std::string& env_str) {
    std::string allowed_specials = ".:,/_-";
    for (size_t i = 0; i < env_str.size(); i++) {
        char c = env_str[i];
        if (!isalnum(c) && (allowed_specials.find(c) == std::string::npos)) {
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

// Create a string representation of the Redis connection
std::string RedisServer::to_string() const
{
    std::string result;

    // Shards
    result += "  Redis shards at:\n";
    auto it = _address_node_map.begin();
    for ( ; it != _address_node_map.end(); it++) {
        result += "    " + it->first + "\n";
    }

    // Protocol
    result += "  Protocol: ";
    result += _is_domain_socket ? "Unix Domain Socket" : "TCP";
    result += "\n";

    // Parameters
    result += "  Command parameters:\n";
    result += "    Retry attempts: "
           + std::to_string(_command_attempts) + "\n";
    result += "    Retry interval (ms): "
           + std::to_string(_command_interval) + "\n";
    result += "    Attempt timeout (ms): "
           + std::to_string(_command_timeout) + "\n";
    result += "  Connection parameters:\n";
    result += "    Retry attempts: "
           + std::to_string(_connection_attempts) + "\n";
    result += "    Retry interval (ms): "
           + std::to_string(_connection_interval) + "\n";
    result += "    Attempt timeout (ms): "
           + std::to_string(_connection_timeout) + "\n";

    // Threadpool
    result += "  Threadpool: " + std::to_string(_thread_count) + " threads\n";

    return result;
}
