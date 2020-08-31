//#ifndef SMARTSIM_CPP_CLIENT_H
#define SMARTSIM_CPP_CLIENT_H
#ifdef __cplusplus
#include "string.h"
#include "stdlib.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <sw/redis++/redis++.h>
#include <fstream>
#include "mpi.h"
///@file
///\brief The C++ SmartSimClient class
class SmartSimClient;

class SmartSimClient
{
public:
  //! SmartSim client constructor
  SmartSimClient(
      bool cluster /*!< Flag to indicate if a database cluster is being used*/,
      bool fortran_array = false /*!< Flag to indicate if fortran arrays are being used*/
  );
  //! SmartSim client destructor
  ~SmartSimClient();

  template <typename T> void get_tensor(std::string_view key, void* result);
  template <class T>void put_tensor(std::string_view key, std::string_view type, int* dims, int n_dims, void* data);
  void run_model(std::string_view key, std::vector<std::string_view> inputs, std::vector<std::string_view> outputs);
  void set_model(std::string_view key, std::string_view backend, std::string_view device, std::string model_file);
  void set_script(std::string_view key, std::string_view device, std::string script_file);
  void run_script(std::string_view key, std::vector<std::string_view> inputs, std::vector<std::string_view> outputs);
  bool key_exists(const char* key);
  bool poll_key(const char* key, int poll_frequency_ms, int num_tries);

protected:
  bool _fortran_array;
  std::string _get_ssdb();
  template <class T>
  void* _add_array_vals_to_buffer(void* data, int* dims, int n_dims, void* buf, int buf_length);
  template <class T>
  void _reformat_data_blob(void* value, int* dims, int n_dims, int& buf_position, void* buf);
  void _put_rai_tensor(std::vector<std::string_view> cmd_args);
  void _get_tensor_dims(std::unique_ptr<redisReply, sw::redis::ReplyDeleter>& reply,
                        int** dims, int& n_dims);
  void _get_tensor_data_blob(std::unique_ptr<redisReply, sw::redis::ReplyDeleter>& reply, void** blob);


private:
  sw::redis::RedisCluster* redis_cluster;
  sw::redis::Redis* redis;

};
#endif //SMARTSIM_CPP_CLIENT_H
