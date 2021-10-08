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

#include <ctype.h>
#include "redisserver.h"

using namespace SmartRedis;

// We should never seed srand more than once. There are more elegant ways
// to prevent it, but this will suffice
static bool ___srand_seeded = false;

// Retrieve a single address, randomly chosen from a list of addresses if
// applicable, from the SSDB environment variable
std::string RedisServer::_get_ssdb()
{
    // Retrieve the environment variable
    char* env_char = getenv("SSDB");
    if (env_char == NULL)
        throw std::runtime_error("The environment variable SSDB "\
                                "must be set to use the client.");
    std::string env_str = std::string(env_char);
    _check_ssdb_string(env_str);

    // Parse the data in it
    std::vector<std::string> hosts_ports;
    const char delim = ',';

    size_t i_pos = 0;
    size_t j_pos = env_str.find(delim);
    while (j_pos != std::string::npos) {
        hosts_ports.push_back("tcp://"+
        env_str.substr(i_pos, j_pos - i_pos));
        i_pos = j_pos + 1;
        j_pos = env_str.find(delim, i_pos);
    }
    // Catch the last value that does not have a trailing ';'
    if (i_pos < env_str.size())
        hosts_ports.push_back("tcp://"+
        env_str.substr(i_pos, j_pos - i_pos));

    // Pick an entry from the list at random, seeding the RNG if needed
    if (!___srand_seeded) {
        std::chrono::high_resolution_clock::time_point t =
            std::chrono::high_resolution_clock::now();

        srand(std::chrono::time_point_cast<std::chrono::nanoseconds>(t).
            time_since_epoch().count());
        ___srand_seeded = true;
    }
    size_t hp = ((size_t)rand()) % hosts_ports.size();

    // Done
    return hosts_ports[hp];
}

// Check that the SSDB environment variable value does not have any errors
void RedisServer::_check_ssdb_string(const std::string& env_str) {
    for (size_t i = 0; i < env_str.size(); i++) {
        char c = env_str[i];
        if (!isalnum(c) && c != '.' && c != ':' && c != ',') {
            throw std::runtime_error("The provided SSDB value, " + env_str +
                                     " is invalid because of character " + c);
        }
    }
}
