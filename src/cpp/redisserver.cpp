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

#include "redisserver.h"

using namespace SmartRedis;

std::string RedisServer::_get_ssdb()
{
    /* This function retrieves the SSDB environment
    variable.  If more than one address is contained
    in SSDB, then one of the addresses is randomly
    selected.
    */
    char* env_char = getenv("SSDB");

    if(!env_char)
        throw std::runtime_error("The environment variable SSDB "\
                                "must be set to use the client.");

    std::string env_str = std::string(env_char);

    this->_check_ssdb_string(env_str);

    std::vector<std::string> hosts_ports;

    const char delim = ',';

    size_t i_pos = 0;
    size_t j_pos = env_str.find(delim);
    while(j_pos!=std::string::npos) {
        hosts_ports.push_back("tcp://"+
        env_str.substr(i_pos, j_pos-i_pos));
        i_pos = j_pos + 1;
        j_pos = env_str.find(delim, i_pos);
    }
    //Catch the last value that does not have a trailing ';'
    if(i_pos<env_str.size())
        hosts_ports.push_back("tcp://"+
        env_str.substr(i_pos, j_pos-i_pos));

    std::chrono::high_resolution_clock::time_point t =
        std::chrono::high_resolution_clock::now();

    srand(std::chrono::time_point_cast<std::chrono::nanoseconds>(t).time_since_epoch().count());
    int hp = rand()%hosts_ports.size();

    return hosts_ports[hp];
}

void RedisServer::_check_ssdb_string(const std::string& env_str) {

  /* This function checks that the ssdb string
  only contains permissable characters, and if a
  character is not allowed, an error will be thrown.
  */

  char c;
  for(size_t i=0; i<env_str.size(); i++) {
      c = env_str[i];
      if( !(c>='0'&& c<='9') &&
          !(c>='a'&& c<='z') &&
          !(c>='A'&& c<='Z') &&
          !(c=='.') &&
          !(c==':') &&
          !(c==',') ) {
            throw std::runtime_error("The provided SSDB value, "
                                     + env_str +
                                     " is not valid because of "\
                                     "character " + c);
          }
  }
  return;
}