#include "redisserver.h"

using namespace SILC;

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