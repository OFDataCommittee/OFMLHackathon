#include "client.h"

SmartSimClient::SmartSimClient(bool cluster, bool fortran_array)
{
  this->_fortran_array = fortran_array;
  if (cluster)
  {
    redis_cluster = new sw::redis::RedisCluster(_get_ssdb());
    redis = 0;
  }
  else
  {
    redis_cluster = 0;
    redis = new sw::redis::Redis(_get_ssdb());
  }
  return;
}

SmartSimClient::~SmartSimClient() {}

std::string SmartSimClient::_get_ssdb()
{
  char *host_and_port = getenv("SSDB");

  if (host_and_port == NULL)
    throw std::runtime_error("The environment variable SSDB must be set to use the client.");

  std::string ssdb("tcp://");
  ssdb.append(host_and_port);
  return ssdb;
}


void SmartSimClient::run_script(std::string_view key, std::vector<std::string_view> inputs, std::vector<std::string_view> outputs)
{
  //This function will run a RedisAI model.  However, we can only do one input and one output,
  //which we need to fix.  We should look into how the python object clients convert it to
  //c code.

  std::unique_ptr<redisReply, sw::redis::ReplyDeleter> reply;
  std::vector<std::string_view> cmd_args;
  cmd_args.push_back("AI.SCRIPTRUN");
  cmd_args.push_back(key);

  cmd_args.push_back("INPUTS");
  for (int i = 0; i < inputs.size(); i++)
    cmd_args.push_back(inputs[i]);

  cmd_args.push_back("OUTPUTS");
  for (int i = 0; i < outputs.size(); i++)
    cmd_args.push_back(outputs[i]);

  reply = redis->command(cmd_args.begin(), cmd_args.end());
}

void SmartSimClient::run_model(std::string_view key, std::vector<std::string_view> inputs, std::vector<std::string_view> outputs)
{
  //This function will run a RedisAI model.  However, we can only do one input and one output,
  //which we need to fix.  We should look into how the python object clients convert it to
  //c code.

  std::unique_ptr<redisReply, sw::redis::ReplyDeleter> reply;
  std::vector<std::string_view> cmd_args;
  cmd_args.push_back("AI.MODELRUN");
  cmd_args.push_back(key);

  cmd_args.push_back("INPUTS");
  for (int i = 0; i < inputs.size(); i++)
    cmd_args.push_back(inputs[i]);

  cmd_args.push_back("OUTPUTS");
  for (int i = 0; i < outputs.size(); i++)
    cmd_args.push_back(outputs[i]);

  if (redis_cluster)
    reply = redis_cluster->command(cmd_args.begin(), cmd_args.end());
  else
    reply = redis->command(cmd_args.begin(), cmd_args.end());
}

void SmartSimClient::set_model(std::string_view key, std::string_view backend, std::string_view device, std::string model_file)
{
  //This function will set a model that is saved in a file.  It is assumed it is a binary file.
  //TODO need to examine code of optimum read to reduce time and memory
  //We really should read into a c str if we are goign to just make a string view at the end.
  std::ifstream fin(model_file.c_str(), std::ios::binary);
  std::ostringstream ostream;
  ostream << fin.rdbuf();

  const std::string tmp = ostream.str();
  const char *model_buf = tmp.c_str();
  int model_buf_len = tmp.length();

  std::vector<std::string_view> cmd_args;
  cmd_args.push_back("AI.MODELSET");
  cmd_args.push_back(key);
  cmd_args.push_back(backend);
  cmd_args.push_back(device);
  cmd_args.push_back("BLOB");
  cmd_args.push_back(std::string_view(model_buf, model_buf_len));

  std::unique_ptr<redisReply, sw::redis::ReplyDeleter> reply;
  if (redis_cluster)
    reply = redis_cluster->command(cmd_args.begin(), cmd_args.end());
  else
    reply = redis->command(cmd_args.begin(), cmd_args.end());
}

void SmartSimClient::set_script(std::string_view key, std::string_view device, std::string script_file)
{
  //This function will set a script that is saved in a file.  It is assumed it is a ascii file.
  //TODO need to examine code of optimum read to reduce time and memory
  //We really should read into a c str if we are goinng to just make a string view at the end.
  std::ifstream fin(script_file.c_str());
  std::ostringstream ostream;
  ostream << fin.rdbuf();

  const std::string tmp = ostream.str();
  const char *script_buf = tmp.c_str();
  int script_buf_len = tmp.length();

  std::vector<std::string_view> cmd_args;
  cmd_args.push_back("AI.SCRIPTSET");
  cmd_args.push_back(key);
  cmd_args.push_back(device);
  cmd_args.push_back("SOURCE");
  cmd_args.push_back(std::string_view(script_buf, script_buf_len));

  std::unique_ptr<redisReply, sw::redis::ReplyDeleter> reply;
  reply = redis->command(cmd_args.begin(), cmd_args.end());
}

template <class T>
void* SmartSimClient::_add_array_vals_to_buffer(void* data, int* dims, int n_dims,
                                 void* buf, int buf_length)
  {
    //TODO we should check at some point that we don't exceed buf length
    //TODO test in multi dimensions with each dimension having a different set of values (make sure
    // (dimensions are independent))
    if(n_dims > 1) {
        T** current = (T**) data;
        for(int i = 0; i < dims[0]; i++) {
          buf = this->_add_array_vals_to_buffer<T>(*current, &dims[1], n_dims-1, buf, buf_length);
          current++;
        }
    }
    else {
        memcpy(buf, data, sizeof(T)*dims[0]);
        return &((T*)buf)[dims[0]];
    }
    return buf;
  }

void SmartSimClient::_put_rai_tensor(std::vector<std::string_view> cmd_args)
{
  int n_trials = 5;
  bool success = true;
  std::string key("<key placeholder>");
  while (n_trials > 0)
  {
    try
    {
      if (redis_cluster)
      {
        redis_cluster->command(cmd_args.begin(), cmd_args.end());
      }
      else
      {
        redis->command(cmd_args.begin(), cmd_args.end());
      }
      n_trials = -1;
    }
    catch (sw::redis::IoError &e)
    {
      n_trials--;
      std::cout << "WARNING: Caught redis IOError: " << e.what() << std::endl;
      std::cout << "WARNING: Could not set key " << key << " in database. " << n_trials << " more trials will be made." << std::endl;
    }
  }
  if (n_trials == 0)
    throw std::runtime_error("Could not set " + std::string(key) + " in database due to redis IOError.");

  if (!success)
    throw std::runtime_error("Redis failed to receive key: " + std::string(key));

  return;
}


template <class T>
void SmartSimClient::put_tensor(std::string_view key, std::string_view type, int* dims, int n_dims, void* data)
{
  // Calculate the amount of memory to allocate for the data buffer
  // TODO if the n_dims==1 we should just pass through the array without extra
  // memory allocation

  //TODO add check to return if n_dims = 0 and/or dims[0] == 0
  //This should throw and error if anyting is 0
  int n_bytes = 1;
  for (int i = 0; i < n_dims; i++) {
    n_bytes *= dims[i];
  }
  n_bytes *= sizeof(T);

  T* buf = (T*)malloc(n_bytes);

  SmartSimClient::_add_array_vals_to_buffer<T>(data, dims, n_dims, (void*)buf, n_bytes);

  std::vector<std::string_view> cmd_args;
  cmd_args.push_back("AI.TENSORSET");
  cmd_args.push_back(key);
  cmd_args.push_back(type);

  std::vector<std::string> dim_strings;
  for(int i = 0; i < n_dims; i++) {
    dim_strings.push_back(std::to_string(dims[i]));
  }

  for(int i = 0; i < n_dims; i++) {
    cmd_args.push_back(std::string_view(dim_strings[i].c_str(), dim_strings[i].length()));
  }

  cmd_args.push_back("BLOB");
  cmd_args.push_back(std::string_view((char*)buf, n_bytes));

  //std::cout<<"Cmd args:"<<std::endl;
  //for(int i = 0; i < cmd_args.size(); i++)
  //  std::cout<<cmd_args[i]<<std::endl;

  this->_put_rai_tensor(cmd_args);

  delete[] buf;

  return;
}

void SmartSimClient::_get_tensor_dims(std::unique_ptr<redisReply, sw::redis::ReplyDeleter>& reply,
                                      int** dims, int& n_dims)
{
  //This function will fill dims and set n_dims to the number of dimensions that are in the
  //tensor get reply

  //We are going to assume right now that the meta data reply is always in the same order
  //and we are going to just index into the base reply.

  if(reply->elements < 6)
    throw std::runtime_error("The message does not have the correct number of fields");

  redisReply* r = reply->element[3];

  n_dims = r->elements;
  (*dims) = new int[n_dims];

  for(int i = 0; i < r->elements; i++)
  {
    redisReply* r_dim = r->element[i];
    (*dims)[i] = r_dim->integer;
  }

  return;
}

void SmartSimClient::_get_tensor_data_blob(std::unique_ptr<redisReply, sw::redis::ReplyDeleter>& reply,
                                           void** blob)
{
  //This function will fill dims and set n_dims to the number of dimensions that are in the
  //tensor get reply

  //We are going to assume right now that the meta data reply is always in the same order
  //and we are going to just index into the base reply.

  if(reply->elements < 6)
    throw std::runtime_error("The message does not have the correct number of fields");

  redisReply* r = reply->element[5];

  (*blob) = malloc(r->len);
  memcpy(*blob, r->str, r->len);

  return;
}

template <class T>
void SmartSimClient::_reformat_data_blob(void* value, int* dims, int n_dims, int& buf_position, void* buf)
{
  if(n_dims > 1) {
    T** current = (T**) value;
    for(int i = 0; i < dims[0]; i++) {
      this->_reformat_data_blob<T>(*current, &dims[1], n_dims-1, buf_position, buf);
      current++;
    }
  }
  else {
    T* buf_to_copy = &(((T*)buf)[buf_position]);
    memcpy(value, buf_to_copy, dims[0]*sizeof(T));
    buf_position += dims[0];
  }
  return;
}


template <typename T>
void SmartSimClient::get_tensor(std::string_view key, void* result)
{

  int n_trials = 5;
  //Make sure no memory leaks
  std::unique_ptr<redisReply, sw::redis::ReplyDeleter> reply;

  while(n_trials > 0) {
    try {
      if(redis_cluster)
        reply = redis_cluster->command("AI.TENSORGET", key, "META", "BLOB");
      else
        reply = redis->command("AI.TENSORGET", key, "META", "BLOB");
      n_trials = -1;
    }
    catch (sw::redis::IoError& e) {
      n_trials--;
      std::cout<<"WARNING: Caught redis IOError: "<<e.what()<<std::endl;
      std::cout<<"WARNING: Could not get key "<<key<<" from database. "<<n_trials<<" more trials will be made."<<std::endl;
    }
  }
  if(n_trials == 0)
    throw std::runtime_error("Could not retreive "+std::string(key)+" from database due to redis IOError.");

  //This is a quick object inspection of the object.  Note that for
  //   messages we should check the type and not just print out strings.
  //   There is int, double, and string types possible in the message.
  //Need check for if the reply is valid.
  /*
  std::cout<<"The get key is "<<key<<std::endl;
  std::cout<<"The base reply string length is: "<<reply->len<<std::endl;
  std::cout<<"The base type is "<<reply->type<<std::endl;
  if(reply->str)
    std::cout<<"The base reply string is: "<<reply->str<<std::endl;
  for(int i = 0; i < reply->elements; i++) {
      std::cout<<"*** Looking at sub element "<<i<<std::endl;
      redisReply* r = reply->element[i];
      std::cout<<"The sub element type is "<<r->type<<std::endl;
      std::cout<<"The sub reply string length is: "<<r->len<<std::endl;
      if(r->str)
        std::cout<<"The sub reply string is: "<<r->str<<std::endl;
      if(r->elements > 0) {
        for(int i = 0; i < r->elements; i++) {
          std::cout<<"*** *** Looking at subsub element "<<i<<std::endl;
          std::cout<<"subsub element type is "<<r->type<<std::endl;
          redisReply* rr = r->element[i];
          std::cout<<"The subsub reply string length is: "<<rr->len<<std::endl;
          if(rr->str)
            std::cout<<"The subsub reply string is: "<<rr->str<<std::endl;
        }
      }

  }
  */
/* For now we are only interested in the BLOB and will return the blob message without
  checking size or dimensions (assuming 1D for for).  We will return a raw pointer to that address.

*/
  void* binary_data = 0;
  int* dims = 0;
  int n_dims = 0;
  int pos = 0;
  this->_get_tensor_dims(reply, &dims, n_dims);
  this->_get_tensor_data_blob(reply, &binary_data);
  this->_reformat_data_blob<T>(result, dims, n_dims, pos, binary_data);

  delete[] dims;
}
// These routines potentially modify keys by adding a prefix
bool SmartSimClient::key_exists(const char* key)
{
  if(redis_cluster)
    return redis_cluster->exists(std::string_view(key));
  else
    return redis->exists(std::string_view(key));
}

bool SmartSimClient::poll_key(const char* key, int poll_frequency_ms, int num_tries)
{
  bool key_exists = false;

  while(!(num_tries==0)) {
    if(this->key_exists(key)) {
      key_exists = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(poll_frequency_ms));
    if(num_tries>0)
      num_tries--;
  }

  if(key_exists)
    return true;
  else
    return false;
}

