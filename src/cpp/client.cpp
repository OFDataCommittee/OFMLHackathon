#include "client.h"

using namespace SILC;

Client::Client(bool cluster, bool fortran_array)
{
  /* Client constructor
  */
  this->_fortran_array = fortran_array;

  int n_connection_trials = 10;

  while(n_connection_trials > 0) {
    if(cluster) {
      try {
        redis_cluster = new sw::redis::RedisCluster(_get_ssdb());
        n_connection_trials = -1;
        redis = 0;
      }
      catch (sw::redis::TimeoutError &e) {
        std::cout << "WARNING: Caught redis TimeoutError: "
                  << e.what() << std::endl;
        std::cout << "WARNING: TimeoutError occurred with "\
                     "initial client connection.";
        std::cout << "WARNING: "<< n_connection_trials
                  << " more trials will be made.";
        n_connection_trials--;
        std::this_thread::sleep_for(std::chrono::seconds(2));
      }

    }
    else {
      try {
        redis = new sw::redis::Redis(_get_ssdb());
        n_connection_trials = -1;
      }
      catch (sw::redis::TimeoutError &e) {
        std::cout << "WARNING: Caught redis TimeoutError: "
                  << e.what() << std::endl;
        std::cout << "WARNING: TimeoutError occurred with "\
                     "initial client connection.";
        std::cout << "WARNING: "<< n_connection_trials
                  << " more trials will be made.";
        n_connection_trials--;
        std::this_thread::sleep_for(std::chrono::seconds(2));
      }
      redis_cluster = 0;
      this->_populate_db_node_data(false);
    }
  }

  if(n_connection_trials==0)
    throw std::runtime_error("A connection could not be "\
                             "established to the redis database");

  if(cluster)
    this->_populate_db_node_data(true);
  else
    this->_populate_db_node_data(false);

  this->_set_prefixes_from_env();

  return;
}

Client::~Client() {
  delete this->redis_cluster;
  delete this->redis;
}

void Client::put_dataset(DataSet& dataset)
{
  /* This function will place a DataSet into the database
  */
  CommandList cmds;
  Command* cmd;

  // Send the metadata message
  cmd = cmds.add_command();
  cmd->add_field("SET");
  cmd->add_field(this->_put_prefix() + "{" +
                 dataset.name + "}" + ".meta");
  cmd->add_field_ptr(dataset.get_metadata_buf());

  // Send the tensor data
  DataSet::tensor_iterator it = dataset.tensor_begin();
  DataSet::tensor_iterator it_end = dataset.tensor_end();
  TensorBase* tensor = 0;
  while(it != it_end) {
    tensor = *it;
    cmd = cmds.add_command();
    cmd->add_field("AI.TENSORSET");
    cmd->add_field(this->_put_prefix() + "{"
                   + dataset.name + "}."
                   + tensor->name());
    cmd->add_field(tensor->type_str());
    cmd->add_fields(tensor->dims());
    cmd->add_field("BLOB");
    cmd->add_field_ptr(tensor->buf());
    it++;
  }
  this->_execute_commands(cmds);
  return;
}

DataSet Client::get_dataset(const std::string& name)
{
  /* This function will retrieve a dataset from the database
  */

  // Get the metadata message and construct DataSet
  CommandReply reply;
  Command cmd;
  cmd.add_field("GET");
  cmd.add_field(this->_get_prefix() +
                "{" + std::string(name) +
                "}" + ".meta");
  reply =  this->_execute_command(cmd);

  DataSet dataset = DataSet(name, reply.str(), reply.str_len());

  // Loop through and add each tensor to the dataset
  std::vector<std::string> tensor_names =
    dataset.get_meta_strings(".tensors");

  for(size_t i=0; i<tensor_names.size(); i++) {
    std::string t_name = this->_get_prefix() +
                         "{" + dataset.name + "}."
                             + tensor_names[i];
    Command cmd;
    cmd.add_field("AI.TENSORGET");
    cmd.add_field(t_name);
    cmd.add_field("META");
    cmd.add_field("BLOB");
    CommandReply reply;
    reply = this->_execute_command(cmd);

    std::vector<size_t> reply_dims = this->_get_tensor_dims(reply);
    std::string_view blob = this->_get_tensor_data_blob(reply);
    TensorType type = this->_get_tensor_data_type(reply);
    dataset._add_to_tensorpack(tensor_names[i],
                               (void*)blob.data(), reply_dims,
                               type, MemoryLayout::contiguous);
  }
  return dataset;
}

void Client::rename_dataset(const std::string& name,
                            const std::string& new_name)
{
  /* This function will rename a dataset (all tensors
  and metadata).
  */
  throw std::runtime_error("rename_dataset is incomplete");
  return;
}

void Client::copy_dataset(const std::string& src_name,
                          const std::string& dest_name)
{
  /* This function will copy a dataset (all tensors
  and metadata).
  */
  throw std::runtime_error("copy_dataset is incomplete");
  return;
}

void Client::delete_dataset(const std::string& name)
{
  /* This function will delete a dataset (all tensors
  and metadata).
  */
  throw std::runtime_error("delete_dataset is incomplete");
  return;
}

void Client::put_tensor(const std::string& key,
                        void* data,
                        const std::vector<size_t>& dims,
                        const TensorType type,
                        const MemoryLayout mem_layout)
{
  /* This function puts a tensor into the datastore
  */

  TensorBase* tensor;

  switch(type) {
    case TensorType::dbl : {
      tensor = new Tensor<double>(key, data, dims,
                                  type, mem_layout);
    break;
    }
    case TensorType::flt :
      tensor = new Tensor<float>(key, data, dims,
                                 type, mem_layout);
    break;
    case TensorType::int64 :
      tensor = new Tensor<int64_t>(key, data, dims,
                                   type, mem_layout);
    break;
    case TensorType::int32 :
      tensor = new Tensor<int32_t>(key, data, dims,
                                   type, mem_layout);
    break;
    case TensorType::int16 :
      tensor = new Tensor<int16_t>(key, data, dims,
                                   type, mem_layout);
    break;
    case TensorType::int8 :
      tensor = new Tensor<int8_t>(key, data, dims,
                                  type, mem_layout);
    break;
    case TensorType::uint16 :
      tensor = new Tensor<uint16_t>(key, data, dims,
                                    type, mem_layout);
    break;
    case TensorType::uint8 :
      tensor = new Tensor<uint8_t>(key, data, dims,
                                   type, mem_layout);
    break;
  }

  CommandReply reply;
  Command cmd;

  cmd.add_field("AI.TENSORSET");
  cmd.add_field(this->_put_prefix() + key);
  cmd.add_field(tensor->type_str());
  for(size_t i=0; i<dims.size(); i++)
    cmd.add_field(std::to_string(dims[i]));
  cmd.add_field("BLOB");
  cmd.add_field_ptr(tensor->buf());
  reply = this->_execute_command(cmd);

  delete tensor;
}

void Client::get_tensor(const std::string& key,
                        void*& data,
                        std::vector<size_t>& dims,
                        TensorType& type,
                        const MemoryLayout mem_layout)
{
  /* This function gets a tensor from the database,
  allocates memory in the specified format for the
  user, sets the dimensions of the dims vector
  for the user, and points the data pointer to
  the allocated memory space.
  */
  CommandReply reply;
  Command cmd;

  cmd.add_field("AI.TENSORGET");
  cmd.add_field(this->_get_prefix() + key);
  cmd.add_field("META");
  cmd.add_field("BLOB");
  reply = this->_execute_command(cmd);

  dims = this->_get_tensor_dims(reply);
  type = this->_get_tensor_data_type(reply);
  std::string_view blob = this->_get_tensor_data_blob(reply);

  if(dims.size()<=0)
    throw std::runtime_error("The number of dimensions of the fetched "\
                             "tensor are invalid: " +
                             std::to_string(dims.size()));

  for(size_t i=0; i<dims.size(); i++) {
    if(dims[i]<=0) {
      throw std::runtime_error("Dimension " + std::to_string(dims[i]) +
                               "of the fetched tensor is not valid: " +
                               std::to_string(dims[i]));
    }
  }

  TensorBase* ptr;
  switch(type) {
    case TensorType::dbl :
      ptr = new Tensor<double>(key, (void*)blob.data(), dims,
                               type, MemoryLayout::contiguous);
      this->_tensor_memory.add_tensor(ptr);
      break;
    case TensorType::flt :
      ptr = new Tensor<float>(key, (void*)blob.data(), dims,
                              type, MemoryLayout::contiguous);
      this->_tensor_memory.add_tensor(ptr);
      break;
    case TensorType::int64 :
      ptr = new Tensor<int64_t>(key, (void*)blob.data(), dims,
                                type, MemoryLayout::contiguous);
      this->_tensor_memory.add_tensor(ptr);
      break;
    case TensorType::int32 :
      ptr = new Tensor<int32_t>(key, (void*)blob.data(), dims,
                                type, MemoryLayout::contiguous);
      this->_tensor_memory.add_tensor(ptr);
      break;
    case TensorType::int16 :
      ptr = new Tensor<int16_t>(key, (void*)blob.data(), dims,
                                type, MemoryLayout::contiguous);
      this->_tensor_memory.add_tensor(ptr);
      break;
    case TensorType::int8 :
      ptr = new Tensor<int8_t>(key, (void*)blob.data(), dims,
                               type, MemoryLayout::contiguous);
      this->_tensor_memory.add_tensor(ptr);
      break;
    case TensorType::uint16 :
      ptr = new Tensor<uint16_t>(key, (void*)blob.data(), dims,
                                 type, MemoryLayout::contiguous);
      this->_tensor_memory.add_tensor(ptr);
      break;
    case TensorType::uint8 :
      ptr = new Tensor<uint8_t>(key, (void*)blob.data(), dims,
                                type, MemoryLayout::contiguous);
      this->_tensor_memory.add_tensor(ptr);
      break;
    default :
      throw std::runtime_error("The tensor could not be retrieved "\
                               "in client.get_tensor().");
      break;
  }
  data = ptr->data_view(mem_layout);
  return;
}

void Client::get_tensor(const std::string& key,
                                void*& data,
                                size_t*& dims,
                                size_t& n_dims,
                                TensorType& type,
                                const MemoryLayout mem_layout)
{
  /* This function will retrieve tensor data
  pointer to the user.  This interface is a c-style
  interface for the dimensions and type for the
  c-interface because memory mamange for
  the dimensions needs to be handled within
  the client.  The retrieved tensor data and
  dimensions will be freed when the client
  is destroyed.
  */
  std::vector<size_t> dims_vec;
  this->get_tensor(key, data, dims_vec,
                   type, mem_layout);

  size_t dims_bytes = sizeof(size_t)*dims_vec.size();
  dims = this->_dim_queries.allocate_bytes(dims_bytes);
  n_dims = dims_vec.size();

  std::vector<size_t>::const_iterator it = dims_vec.cbegin();
  std::vector<size_t>::const_iterator it_end = dims_vec.cend();
  size_t i = 0;
  while(it!=it_end) {
      dims[i] = *it;
      i++;
      it++;
  }

  return;
}

void Client::unpack_tensor(const std::string& key,
                          void* data,
                          const std::vector<size_t>& dims,
                          const TensorType type,
                          const MemoryLayout mem_layout)
{
  /* This function gets a tensor from the database and
  copies the data to the data pointer location.
  */

  if(mem_layout == MemoryLayout::contiguous &&
     dims.size()>1) {
       throw std::runtime_error("The destination memory space "\
                                "dimension vector should only "\
                                "be of size one if the memory "\
                                "layout is contiguous.");
     }

  CommandReply reply;
  Command cmd;

  cmd.add_field("AI.TENSORGET");
  cmd.add_field(this->_get_prefix() + key);
  cmd.add_field("META");
  cmd.add_field("BLOB");
  reply = this->_execute_command(cmd);

  std::vector<size_t> reply_dims = this->_get_tensor_dims(reply);

  if(mem_layout == MemoryLayout::contiguous ||
     mem_layout == MemoryLayout::fortran_contiguous) {

    int total_dims = 1;
    for(size_t i=0; i<reply_dims.size(); i++) {
      total_dims *= reply_dims[i];
    }
    if( (total_dims != dims[0]) && (mem_layout == MemoryLayout::contiguous) )
      throw std::runtime_error("The dimensions of the fetched tensor "\
                               "do not match the length of the "\
                               "contiguous memory space.");
  }

  if(mem_layout == MemoryLayout::nested) {
    if(dims.size()!= reply_dims.size())
      throw std::runtime_error("The number of dimensions of the  "\
                               "fetched tensor, " +
                               std::to_string(reply_dims.size()) + " "\
                               "does not match the number of dimensions "\
                               "of the user memory space, " +
                               std::to_string(dims.size()));

    for(size_t i=0; i<reply_dims.size(); i++) {
      if(dims[i]!=reply_dims[i]) {
        throw std::runtime_error("The dimensions of the fetched tensor "\
                                 "do not match the provided dimensions "\
                                 "of the user memory space.");
      }
    }
  }

  TensorType reply_type = this->_get_tensor_data_type(reply);
  if(type!=reply_type)
    throw std::runtime_error("The type of the fetched tensor "\
                             "does not match the provided type");
  std::string_view blob = this->_get_tensor_data_blob(reply);

  TensorBase* tensor;
  switch(reply_type) {
    case TensorType::dbl :
      tensor = new Tensor<double>(key, (void*)blob.data(),
                                  reply_dims, reply_type,
                                  MemoryLayout::contiguous);
    break;
    case TensorType::flt :
      tensor = new Tensor<float>(key, (void*)blob.data(),
                                 reply_dims, reply_type,
                                 MemoryLayout::contiguous);
    break;
    case TensorType::int64  :
      tensor = new Tensor<int64_t>(key, (void*)blob.data(),
                                   reply_dims, reply_type,
                                   MemoryLayout::contiguous);
    break;
    case TensorType::int32 :
      tensor = new Tensor<int32_t>(key, (void*)blob.data(),
                                   reply_dims, reply_type,
                                   MemoryLayout::contiguous);
    break;
    case TensorType::int16 :
      tensor = new Tensor<int16_t>(key, (void*)blob.data(),
                                   reply_dims, reply_type,
                                   MemoryLayout::contiguous);
    break;
    case TensorType::int8 :
      tensor = new Tensor<int8_t>(key, (void*)blob.data(),
                                  reply_dims, reply_type,
                                  MemoryLayout::contiguous);
    break;
    case TensorType::uint16 :
      tensor = new Tensor<uint16_t>(key, (void*)blob.data(),
                                    reply_dims, reply_type,
                                    MemoryLayout::contiguous);
    break;
    case TensorType::uint8 :
      tensor = new Tensor<uint8_t>(key, (void*)blob.data(),
                                   reply_dims, reply_type,
                                   MemoryLayout::contiguous);
    break;
  }

  tensor->fill_mem_space(data, dims, mem_layout);
  delete tensor;
  return;
}

void Client::rename_tensor(const std::string& key,
                                   const std::string& new_key)
{
  /* This function moves a tensor from the key to the new_key.
  If the db is a redis cluster and the two keys hash to
  different nodes, copy and delete operations are performed,
  otherwise a single move command is used.
  */
  uint16_t key_slot = 0;
  uint16_t new_key_slot = 0;
  if(this->redis_cluster) {
    key_slot = this->_get_hash_slot(key);
    new_key_slot = this->_get_hash_slot(new_key);
  }

  if(this->redis_cluster && key_slot!=new_key_slot) {
    this->copy_tensor(key, new_key);
    this->delete_tensor(key);
  }
  else {
    CommandReply reply;
    Command cmd;
    cmd.add_field("MOVE");
    cmd.add_field(key);
    cmd.add_field(new_key);
    reply = this->_execute_command(cmd);
  }
  return;
}

void Client::delete_tensor(const std::string& key)
{
  /* This function will delete a tensor from the database
  */
  CommandReply reply;
  Command cmd;
  cmd.add_field("DEL");
  cmd.add_field(key);

  std::string hash_tag;
  if(this->_has_hash_tag(key))
    hash_tag = this->_get_hash_tag(key);

  if(hash_tag.size()>0)
    reply = this->_execute_command(cmd, hash_tag);
  else
    reply = this->_execute_command(cmd);
  return;
}

void Client::copy_tensor(const std::string& src_key,
                         const std::string& dest_key)
{
  /* This function copies a tensor from the src_key to the
  dest_key.
  */
  this->_copy_tensor(src_key, dest_key);
  return;
}

void Client::set_model_from_file(const std::string& key,
                                 const std::string& model_file,
                                 const std::string& backend,
                                 const std::string& device,
                                 int batch_size,
                                 int min_batch_size,
                                 const std::string& tag,
                                 const std::vector<std::string>& inputs,
                                 const std::vector<std::string>& outputs)
{
  /*This function will set a model from the database that is saved
  in a file.  It is assumed that the file is a binary file.
  */
  if(model_file.size()==0)
    throw std::runtime_error("model_file is a required  "
                             "parameter of set_model.");

  std::ifstream fin(model_file, std::ios::binary);
  std::ostringstream ostream;
  ostream << fin.rdbuf();

  const std::string tmp = ostream.str();
  std::string_view model(tmp.data(), tmp.length());

  this->set_model(key, model, backend, device, batch_size,
                  min_batch_size, tag, inputs, outputs);
  return;
}

void Client::set_model(const std::string& key,
                       const std::string_view& model,
                       const std::string& backend,
                       const std::string& device,
                       int batch_size,
                       int min_batch_size,
                       const std::string& tag,
                       const std::vector<std::string>& inputs,
                       const std::vector<std::string>& outputs)
{
  /*This function will set a model from the provided buffer.
  */
  if(key.size()==0)
    throw std::runtime_error("key is a required parameter "
                             "of set_model.");

  if(backend.size()==0)
    throw std::runtime_error("backend is a required  "
                             "parameter of set_model.");

  if(backend.compare("TF")!=0) {
    if(inputs.size() > 0)
      throw std::runtime_error("INPUTS in the model set command "\
                               "is only valid for TF models");
    if(outputs.size() > 0)
      throw std::runtime_error("OUTPUTS in the model set command "\
                               "is only valid for TF models");
  }

  if(backend.compare("TF")!=0 && backend.compare("TFLITE")!=0 &&
     backend.compare("TORCH")!=0 && backend.compare("ONNX")!=0) {
      throw std::runtime_error(std::string(backend) +
                                " is not a valid backend.");
  }

  if(device.size()==0)
    throw std::runtime_error("device is a required  "
                             "parameter of set_model.");

  if(device.compare("CPU")!=0 &&
     std::string(device).find("GPU")==std::string::npos) {
       throw std::runtime_error(std::string(backend) +
                                " is not a valid backend.");
  }

  std::string prefixed_key;
  if(this->redis_cluster) {
    std::vector<DBNode>::const_iterator node =
      this->_db_nodes.cbegin();
    std::vector<DBNode>::const_iterator end_node =
      this->_db_nodes.cend();
    while(node!=end_node)
    {
      prefixed_key = "{" + node->prefix +
                     "}." + std::string(key);
      this->_set_model(prefixed_key, model, backend,
                       device, batch_size, min_batch_size,
                       tag, inputs, outputs);
      node++;
    }
  }
  else {
    prefixed_key = std::string(key);
    this->_set_model(prefixed_key, model, backend,
                     device, batch_size, min_batch_size,
                     tag, inputs, outputs);
  }
  return;
}

std::string_view Client::get_model(const std::string& key)
{
  /* This function returns the model via the char* model
  variable and the length reference variable.  Note that
  the char* model pointer can contain null characters and
  should be used only in conjunction with the length
  variable.
  */
  std::string prefixed_str;
  if(this->redis_cluster) {
    prefixed_str = "{" + this->_db_nodes[0].prefix +
                   "}." + std::string(key);
  }
  else {
    prefixed_str = std::string(key);
  }

  Command cmd;
  CommandReply reply;
  cmd.add_field("AI.MODELGET");
  cmd.add_field(prefixed_str);
  cmd.add_field("BLOB");
  reply = this->_execute_command(cmd);

  char* model = this->_model_queries.allocate(reply.str_len());
  std::memcpy(model, reply.str(), reply.str_len());
  return std::string_view(model, reply.str_len());
}

void Client::set_script_from_file(const std::string& key,
                                  const std::string& device,
                                  const std::string& script_file)
{
  /*This function will set a script that is saved in a file.
  It is assumed it is an ascii file.
  */
  std::ifstream fin(script_file);
  std::ostringstream ostream;
  ostream << fin.rdbuf();

  const std::string tmp = ostream.str();
  std::string_view script(tmp.data(), tmp.length());

  this->set_script(key, device, script);
  return;
}


void Client::set_script(const std::string& key,
                        const std::string& device,
                        const std::string_view& script)
{
  /*This function will set a script from the provided buffer.
  */
  std::string prefixed_key;
  if(this->redis_cluster) {
    std::vector<DBNode>::const_iterator node =
      this->_db_nodes.cbegin();
    std::vector<DBNode>::const_iterator end_node =
      this->_db_nodes.cend();
    while(node!=end_node) {
      prefixed_key = "{" + node->prefix +
                     "}." + std::string(key);
      this->_set_script(prefixed_key, device, script);
      node++;
    }
  }
  else {
    prefixed_key = std::string(key);
    this->_set_script(prefixed_key, device, script);
  }
  return;
}

std::string_view Client::get_script(const std::string& key)
{
  /* This function returns the script via the char* script
  variable and the length reference variable.  Note that
  the char* script pointer can contain null characters and
  should be used only in conjunction with the length
  variable.
  */
  Command cmd;
  std::string prefixed_str;
  if(this->redis_cluster) {
    prefixed_str = "{" + this->_db_nodes[0].prefix +
                   "}." + std::string(key);
  }
  else {
    prefixed_str = std::string(key);
  }

  cmd.add_field("AI.SCRIPTGET");
  cmd.add_field(prefixed_str);
  cmd.add_field("SOURCE");
  CommandReply reply;
  reply = this->_execute_command(cmd);

  char* script = this->_model_queries.allocate(reply.str_len());
  std::memcpy(script, reply.str(), reply.str_len());
  return std::string_view(script, reply.str_len());
}

void Client::run_model(const std::string& key,
                       std::vector<std::string> inputs,
                       std::vector<std::string> outputs)
{
  /*This function will run a RedisAI model.
  */

  //For this version of run model, we have to copy all
  //input and output tensors, so we will randomly select
  //a model.  We can't use rand, because MPI would then
  //have the same random number across all ranks.  Instead
  //We will choose it based on the db of the firs tinput tensor.

  //Update input and output tensor names for ensembling prefix
  this->_append_with_get_prefix(inputs);
  this->_append_with_put_prefix(outputs);

  uint16_t hash_slot = this->_get_hash_slot(inputs[0]);
  uint16_t db_index = this->_get_dbnode_index(hash_slot, 0,
                                              this->_n_db_nodes);
  DBNode* db = &(this->_db_nodes[db_index]);

  //Generate temporary names so that all keys go to same slot
  std::vector<std::string> tmp_inputs =
    _get_tmp_names(inputs, db->prefix);
  std::vector<std::string> tmp_outputs =
    _get_tmp_names(outputs, db->prefix);

  //Copy all input tensors to temporary names to align hash slots
  this->_copy_tensors(inputs, tmp_inputs);

  std::string model_name = "{" + db->prefix +
                           "}." + std::string(key);

  Command cmd;
  cmd.add_field("AI.MODELRUN");
  cmd.add_field(model_name);
  cmd.add_field("INPUTS");
  cmd.add_fields(tmp_inputs);
  cmd.add_field("OUTPUTS");
  cmd.add_fields(tmp_outputs);
  if(this->redis_cluster)
    this->_execute_command(cmd, db->prefix);
  else
    this->_execute_command(cmd);


  this->_copy_tensors(tmp_outputs, outputs);

  std::vector<std::string> keys_to_delete;
  keys_to_delete.insert(keys_to_delete.end(),
                        tmp_outputs.begin(),
                        tmp_outputs.end());
  keys_to_delete.insert(keys_to_delete.end(),
                        tmp_inputs.begin(),
                        tmp_inputs.end());

  this->_delete_keys(keys_to_delete, db->prefix);

  return;
}

void Client::__run_model_dagrun(const std::string& key,
                                std::vector<std::string> inputs,
                                std::vector<std::string> outputs)
{
  /*This function will run a RedisAI model.  Because the RedisAI
  AI.RUNMODEL and AI.DAGRUN commands assume that the tensors
  and model are all on the same node.  As a result, we will
  have to retrieve all input tensors that are not on the same
  node as the model and set temporary
  */

  //TODO We need to make sure that no other clients are using the
  //same keys and model because we may end up overwriting or having
  //race conditions on who can use the model, etc.

  DBNode* db = this->_get_model_script_db(key, inputs, outputs);

  //Create list of input tensors that do not hash to db slots
  std::unordered_set<std::string> remote_inputs;
  for(int i=0; i<inputs.size(); i++) {
    uint16_t hash_slot = this->_get_hash_slot(inputs[i]);
    if(hash_slot < db->lower_hash_slot ||
       hash_slot > db->upper_hash_slot)
       remote_inputs.insert(inputs[i]);
  }

  //Retrieve tensors that do not hash to db,
  //rename the tensors to {prefix}.tensor_name.TMP
  //TODO we need to make sure users don't use the .TMP suffix
  //or check that the key does not exist
  for(int i=0; i<inputs.size(); i++) {
    if(remote_inputs.count(inputs[i])>0) {
      std::string new_key = "{" + db->prefix + "}." +
                            inputs[i] + ".TMP";
      this->copy_tensor(inputs[i], new_key);
      remote_inputs.erase(inputs[i]);
      remote_inputs.insert(new_key);
      inputs[i] = new_key;
    }
  }

  //Create a renaming scheme for output tensor
  std::unordered_map<std::string, std::string> remote_outputs;
  for(int i=0; i<outputs.size(); i++) {
    uint16_t hash_slot = this->_get_hash_slot(outputs[i]);
    if(hash_slot < db->lower_hash_slot ||
       hash_slot > db->upper_hash_slot) {
        std::string tmp_name = "{" + db->prefix + "}." +
                               outputs[i] + ".TMP";
        remote_outputs.insert({outputs[i], tmp_name});
        outputs[i] = remote_outputs[outputs[i]];
    }
  }

  std::string model_name = "{" + db->prefix +
                           "}." + std::string(key);
  Command cmd;

  cmd.add_field("AI.DAGRUN");
  cmd.add_field("LOAD");
  cmd.add_field(std::to_string(inputs.size()));
  cmd.add_fields(inputs);
  cmd.add_field("PERSIST");
  cmd.add_field(std::to_string(outputs.size()));
  cmd.add_fields(outputs);
  cmd.add_field("|>");
  cmd.add_field("AI.MODELRUN");
  cmd.add_field(model_name);
  cmd.add_field("INPUTS");
  cmd.add_fields(inputs);
  cmd.add_field("OUTPUTS");
  cmd.add_fields(outputs);
  if(this->redis_cluster)
    this->_execute_command(cmd, db->prefix);
  else
    this->_execute_command(cmd);

  //Delete temporary input tensors
  std::unordered_set<std::string>::const_iterator i_it
    = remote_inputs.begin();
  std::unordered_set<std::string>::const_iterator i_it_end
    = remote_inputs.end();
  while(i_it!=i_it_end) {
    this->delete_tensor(*i_it);
    i_it++;
  }

  //Move temporary output to the correct location and
  //delete temporary output tensors
  std::unordered_map<std::string, std::string>::const_iterator j_it
    = remote_outputs.begin();
  std::unordered_map<std::string, std::string>::const_iterator j_it_end
    = remote_outputs.end();
  while(j_it!=j_it_end) {
    this->rename_tensor(j_it->second, j_it->first);
    j_it++;
  }

  return;
}

void Client::run_script(const std::string& key,
                        const std::string& function,
                        std::vector<std::string> inputs,
                        std::vector<std::string> outputs)
{
  /*This function will run a RedisAI script.
  */

  //Update input and output tensor names for ensembling prefix
  this->_append_with_get_prefix(inputs);
  this->_append_with_put_prefix(outputs);

  uint16_t hash_slot = this->_get_hash_slot(inputs[0]);
  uint16_t db_index = this->_get_dbnode_index(hash_slot, 0,
                                              this->_n_db_nodes);
  DBNode* db = &(this->_db_nodes[db_index]);

  //Generate temporary names so that all keys go to same slot
  std::vector<std::string> tmp_inputs =
    _get_tmp_names(inputs, db->prefix);
  std::vector<std::string> tmp_outputs =
    _get_tmp_names(outputs, db->prefix);

  //Copy all input tensors to temporary names to align hash slots
  this->_copy_tensors(inputs, tmp_inputs);

  std::string script_name = "{" + db->prefix +
                           "}." + std::string(key);
  Command cmd;

  cmd.add_field("AI.SCRIPTRUN");
  cmd.add_field(script_name);
  cmd.add_field(function);
  cmd.add_field("INPUTS");
  cmd.add_fields(tmp_inputs);
  cmd.add_field("OUTPUTS");
  cmd.add_fields(tmp_outputs);
  if(this->redis_cluster)
    this->_execute_command(cmd, db->prefix);
  else
    this->_execute_command(cmd);

  this->_copy_tensors(tmp_outputs, outputs);

  std::vector<std::string> keys_to_delete;
  keys_to_delete.insert(keys_to_delete.end(),
                        tmp_outputs.begin(),
                        tmp_outputs.end());
  keys_to_delete.insert(keys_to_delete.end(),
                        tmp_inputs.begin(),
                        tmp_inputs.end());

  this->_delete_keys(keys_to_delete, db->prefix);
  return;
}

inline void Client::_set_model(const std::string& model_name,
                               std::string_view model,
                               const std::string& backend,
                               const std::string& device,
                               int batch_size,
                               int min_batch_size,
                               const std::string& tag,
                               const std::vector<std::string>& inputs,
                               const std::vector<std::string>& outputs
                               )
{
  /*This function will set the provided model into the database
  */
  Command cmd;
  cmd.add_field("AI.MODELSET");
  cmd.add_field(model_name);
  cmd.add_field(backend);
  cmd.add_field(device);
  if(tag.size()>0) {
    cmd.add_field("TAG");
    cmd.add_field(tag);
  }
  if(batch_size>0) {
    cmd.add_field("BATCHSIZE");
    cmd.add_field(std::to_string(batch_size));
  }
  if(min_batch_size>0) {
    cmd.add_field("MINBATCHSIZE");
    cmd.add_field(std::to_string(min_batch_size));
  }
  if(inputs.size()>0) {
    cmd.add_field("INPUTS");
    cmd.add_fields(inputs);
  }
  if(outputs.size()>0) {
    cmd.add_field("OUTPUTS");
    cmd.add_fields(outputs);
  }
  cmd.add_field("BLOB");
  cmd.add_field_ptr(model);
  this->_execute_command(cmd);
  return;
}

inline void Client::_set_script(const std::string& script_name,
                                const std::string& device,
                                std::string_view script)
{
  /*This function will set the provided script into the database.
  */
  Command cmd;
  cmd.add_field("AI.SCRIPTSET");
  cmd.add_field(script_name);
  cmd.add_field(device);
  cmd.add_field("SOURCE");
  cmd.add_field_ptr(script);
  this->_execute_command(cmd);
  return;
}

inline void Client::_copy_tensor(const std::string& src_key,
                                 const std::string& dest_key)
{
  /* This function a tensor from src_key
  to dest_key.
  */

  CommandReply cmd_get_reply;
  Command cmd_get;

  cmd_get.add_field("AI.TENSORGET");
  cmd_get.add_field(src_key);
  cmd_get.add_field("META");
  cmd_get.add_field("BLOB");

  std::string get_hash_tag;
  if(this->_has_hash_tag(src_key))
    get_hash_tag = this->_get_hash_tag(src_key);

  cmd_get_reply = this->_execute_command(cmd_get, get_hash_tag);

  std::vector<size_t> dims = this->_get_tensor_dims(cmd_get_reply);
  std::string_view blob = this->_get_tensor_data_blob(cmd_get_reply);
  TensorType type = this->_get_tensor_data_type(cmd_get_reply);

  CommandReply cmd_put_reply;
  Command cmd_put;

  cmd_put.add_field("AI.TENSORSET");
  cmd_put.add_field(dest_key);
  cmd_put.add_field(TENSOR_STR_MAP.at(type));
  for(size_t i=0; i<dims.size(); i++)
    cmd_put.add_field(std::to_string(dims[i]));
  cmd_put.add_field("BLOB");
  cmd_put.add_field_ptr(blob);

  std::string dest_hash_key;
  if(this->_has_hash_tag(dest_key))
    dest_hash_key = this->_get_hash_tag(dest_key);

  cmd_put_reply = this->_execute_command(cmd_put, dest_hash_key);

  return;
}

void Client::_populate_db_node_data(bool cluster)
{
  /*This function retrieves hash slot information from the cluster
  and stores it in the client data structure.  This is currently
  accomplished with a "CLUSTER SLOTS" command but could be
  accomplished later with direct call of redis library
  hash slot assignment subroutines.
  */

  CommandReply reply;
  Command cmd;
  if(cluster) {
    cmd.add_field("CLUSTER");
    cmd.add_field("SLOTS");
    reply = this->_execute_command(cmd);
    this->_parse_reply_for_slots(reply);
  }
  else {
    this->_db_nodes = std::vector<DBNode>(1);
    this->_db_nodes[0].lower_hash_slot = 0;
    this->_db_nodes[0].upper_hash_slot = 16384;
    //Get the node and port
    std::string node_port = this->_get_ssdb();
    std::string::size_type i = node_port.find("tcp://");
    if (i != std::string::npos)
      node_port.erase(i, 6);
    i = node_port.find(":");
    if (i == std::string::npos)
      throw std::runtime_error("SSDB is not formatted correctly.");
    this->_db_nodes[0].ip = node_port.substr(0, i);
    this->_db_nodes[0].port = atoi(node_port.substr(i).c_str());
    this->_db_nodes[0].prefix = this->_get_crc16_prefix(
                                this->_db_nodes[0].lower_hash_slot);
  }
  return;
}

void Client::_parse_reply_for_slots(CommandReply& reply)
{
  /*This function parses a CommandReply for cluster slot
  information.
  Each reply element of the main message, of which there should
  be n_db_nodes, is:
   0) (integer) min slot
   1) (integer) max slot
   2) 0) "ip address"
      1) (integer) port
      2) "name"
  */

  this->_n_db_nodes = reply.n_elements();
  this->_db_nodes = std::vector<DBNode>(this->_n_db_nodes);

  for(int i=0; i<this->_n_db_nodes; i++) {
    this->_db_nodes[i].lower_hash_slot = reply[i][0].integer();
    this->_db_nodes[i].upper_hash_slot = reply[i][1].integer();
    this->_db_nodes[i].ip = std::string(reply[i][2][0].str(),
                                        reply[i][2][0].str_len());
    this->_db_nodes[i].port = reply[i][2][1].integer();
    this->_db_nodes[i].name = std::string(reply[i][2][2].str(),
                                          reply[i][2][2].str_len());
    bool acceptable_prefix = false;
    int n_hashes = this->_db_nodes[i].upper_hash_slot -
                   this->_db_nodes[i].lower_hash_slot + 1;
    int k = 0;
    while(!acceptable_prefix && k<=n_hashes) {
      this->_db_nodes[i].prefix = this->_get_crc16_prefix(
                                  this->_db_nodes[i].lower_hash_slot+k);
      std::string prefix = this->_db_nodes[i].prefix;
      bool found_bracket = false;
      for(int j=0; j<prefix.size(); j++) {
        if(prefix[j] == '}')
          found_bracket = true;
      }
      if(!found_bracket)
        acceptable_prefix=true;
      k++;
    }
    if(k>n_hashes)
      throw std::runtime_error("A prefix could not be generated "\
                              "for this cluster config.");
  }
  //Put the vector of db nodes in order based on lower hash slot
  std::sort(this->_db_nodes.begin(), this->_db_nodes.end());
  return;
}

std::vector<size_t> Client::_get_tensor_dims(CommandReply& reply)
{
  /* This function will fill a vector with the dimensions of the
  tensor.  We assume right now that the META reply is always
  in the same order and we can index base reply elements array;
  */

  if(reply.n_elements() < 6)
    throw std::runtime_error("The message does not have the "\
                             "correct number of fields");

  size_t n_dims = reply[3].n_elements();
  std::vector<size_t> dims(n_dims);

  for(size_t i=0; i<n_dims; i++) {
    dims[i] = reply[3][i].integer();
  }

  return dims;
}

std::string_view Client::_get_tensor_data_blob(CommandReply& reply)
{
  /* Returns a string view of the data tensor blob value
  */

  //We are going to assume right now that the meta data reply
  //is always in the same order and we are going to just
  //index into the base reply.

  if(reply.n_elements() < 6)
    throw std::runtime_error("The message does not have the "\
                            "correct number of fields");

  return std::string_view(reply[5].str(), reply[5].str_len());
}

TensorType Client::_get_tensor_data_type(CommandReply& reply)
{
  /* Returns a string of the tensor type
  */

  //We are going to assume right now that the meta data reply
  //is always in the same order and we are going to just
  //index into the base reply.

  if(reply.n_elements() < 2)
    throw std::runtime_error("The message does not have the correct "\
                             "number of fields");

  return TENSOR_TYPE_MAP.at(std::string(reply[1].str(),
                                        reply[1].str_len()));
}


// These routines potentially modify keys by adding a prefix
bool Client::key_exists(const std::string& key)
{
  std::string_view key_view(key.c_str(), key.size());
  if(redis_cluster)
    return redis_cluster->exists(key_view);
  else
    return redis->exists(key_view);
}

bool Client::poll_key(const std::string& key,
                              int poll_frequency_ms,
                              int num_tries)
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

void Client::_execute_commands(CommandList& cmds,
                               std::string prefix)
{
  /* This function executes a series of Command objects
  contained in a CommandList
  */

  CommandList::iterator cmd = cmds.begin();
  CommandList::iterator cmd_end = cmds.end();

  while(cmd != cmd_end) {
    this->_execute_command(**cmd, prefix);
    cmd++;
  }
}

CommandReply Client::_execute_command(Command& cmd,
                                      std::string prefix)
{
  /* This function executes a Command
  */

  Command::iterator cmd_fields_start = cmd.begin();
  Command::iterator cmd_fields_end = cmd.end();

  CommandReply reply;

  //TODO we need to change the key and success to
  //false when we can check the status of Reply
  int n_trials = 100;
  bool success = true;

  while (n_trials > 0 && success)
  {
    try
    {
      if(prefix.size()>0) {
        const std::string_view sv_prefix(prefix.c_str(), prefix.size());
        sw::redis::Redis server = this->redis_cluster->redis(sv_prefix, false);
        reply = server.command(cmd_fields_start, cmd_fields_end);
      }
      else if (this->redis_cluster) {
        reply = redis_cluster->command(cmd_fields_start, cmd_fields_end);
      }
      else {
        reply = redis->command(cmd_fields_start, cmd_fields_end);
      }

      if(reply.has_error()==0) {
        n_trials = -1;
      }
      else {
        n_trials = 0;
      }
    }
    catch (sw::redis::TimeoutError &e)
    {
      n_trials--;
      std::cout << "WARNING: Caught redis TimeoutError: " << e.what() << std::endl;
      std::cout << "WARNING: Could not execute command " << cmd.first_field()
                << " and " << n_trials << " more trials will be made."
                << std::endl << std::flush;
      std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    catch (sw::redis::IoError &e)
    {
      n_trials--;
      std::cout << "WARNING: Caught redis IOError: " << e.what() << std::endl;
      std::cout << "WARNING: Could not execute command " << cmd.first_field()
                << " and " << n_trials << " more trials will be made."
                << std::endl << std::flush;
    }
    catch (...) {
      n_trials--;
      std::cout << "WARNING: Could not execute command " << cmd.first_field()
                << " and " << n_trials << " more trials will be made."
                << std::endl << std::flush;
      std::cout << "Error command = "<<cmd.to_string()<<std::endl;
      throw;
    }
  }
  if (n_trials == 0)
    success = false;

  if (!success) {
    if(reply.has_error()>0)
      reply.print_reply_error();
    throw std::runtime_error("Redis failed to execute command: " +
                              cmd.to_string());
  }

  return reply;
}

std::string Client::_get_ssdb()
{
  std::string env_str = std::string(getenv("SSDB"));

  if (env_str.size()==0)
    throw std::runtime_error("The environment variable SSDB "\
                             "must be set to use the client.");

  std::vector<std::string> hosts_ports;

  size_t i_pos = 0;
  size_t j_pos = env_str.find(';');
  while(j_pos!=std::string::npos) {
    hosts_ports.push_back("tcp://"+
      env_str.substr(i_pos, j_pos-i_pos));
    i_pos = j_pos + 1;
    j_pos = env_str.find(";", i_pos);
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

uint64_t Client::_crc16_inverse(uint64_t remainder)
{
  /* This function inverts the crc16 process
  to return a message that will result in the
  provided crc16 remainder.
  */
  uint64_t digit = 1;
  uint64_t poly = 69665; //x^16 + x^12 + x^5 + 1

  for(int i=0; i<16; i++) {
    if(remainder&digit)
      remainder = remainder^poly;
    digit=digit<<1;
    poly=poly<<1;
  }
  return remainder;
}

std::string Client::_get_crc16_prefix(uint64_t hash_slot)
{
    /* This takes hash slot and returns a
    two character prefix (potentially with
    null characters)
    */
    uint64_t byte_filter = 255;
    uint64_t crc_out = this->_crc16_inverse(hash_slot);
    crc_out = crc_out >> 16;
    //Get the two character prefix
    char* prefix = new char[2];
    for(int i=1; i>=0; i--) {
        prefix[i] = (crc_out&byte_filter);
        crc_out = crc_out>>8;
    }
    std::string prefix_str = std::string(prefix, 2);
    delete[] prefix;
    return prefix_str;
}

DBNode* Client::_get_model_script_db(const std::string& name,
                                     std::vector<std::string>& inputs,
                                     std::vector<std::string>& outputs)
{
  /* This function calculates the optimal model name to use
  to run the provided inputs.  If a cluster is not being used,
  the model name is returned, else a prefixed model name is returned.
  */

  //TODO we should randomly choose the max if there are multiple
  //maxes
  if(!this->redis_cluster)
    return &(this->_db_nodes[0]);

  std::vector<int> hash_slot_tally(this->_n_db_nodes, 0);

  for(int i=0; i<inputs.size(); i++) {
    uint16_t hash_slot = this->_get_hash_slot(inputs[i]);
    uint16_t db_index = this->_get_dbnode_index(hash_slot, 0,
                                                this->_n_db_nodes);
    hash_slot_tally[db_index]++;
  }

  for(int i=0; i<outputs.size(); i++) {
    uint16_t hash_slot = this->_get_hash_slot(outputs[i]);
    uint16_t db_index = this->_get_dbnode_index(hash_slot, 0,
                                                this->_n_db_nodes);
    hash_slot_tally[db_index]++;
  }

  //Determine which DBNode has the most hashes
  int max_hash = -1;
  DBNode* db = 0;
  for(int i=0; i<this->_n_db_nodes; i++) {
    if(hash_slot_tally[i] > max_hash) {
      max_hash = hash_slot_tally[i];
      db = &(this->_db_nodes[i]);
    }
  }
  return db;
}

bool Client::_has_hash_tag(const std::string& key)
{
  /* This function determines if the key has a hash
  slot.
  */

  size_t first = key.find('{');
  size_t second = key.find('}');
  if(first == std::string::npos ||
     second == std::string::npos)
    return false;
  else if(second < first)
    return false;
  else
    return true;
}

std::string Client::_get_hash_tag(const std::string& key)
{
  /* This function determines if the key has a hash
  slot.
  */

  size_t first = key.find('{');
  size_t second = key.find('}');
  if(first == std::string::npos ||
     second == std::string::npos)
    return key;
  else if(second < first)
    return key;
  else
    return key.substr(first,second-first+1);
}

uint16_t Client::_get_dbnode_index(uint16_t hash_slot,
                                   unsigned lhs, unsigned rhs)
{
  /*  This is a binomial search to determine the
  DBNode that is responsible for the hash slot.
  */
  uint16_t m = (lhs + rhs)/2;
  if(this->_db_nodes[m].lower_hash_slot<=hash_slot &&
     this->_db_nodes[m].upper_hash_slot>=hash_slot) {
       return m;
     }
  else {
    if(this->_db_nodes[m].lower_hash_slot > hash_slot)
      return this->_get_dbnode_index(hash_slot, lhs, m-1);
    else
      return this->_get_dbnode_index(hash_slot, m+1, rhs);
  }
}

uint16_t Client::_get_hash_slot(const std::string& key)
{
  /* This function returns the hash slot of the key.
  */
  std::string hash_key;
  if(this->_has_hash_tag(key))
    hash_key = this->_get_hash_tag(key);
  else
    hash_key = key;
  return sw::redis::crc16(hash_key.c_str(),
                          hash_key.size()) % 16384;
}

void Client::_set_prefixes_from_env()
{
  const char* keyout_p = std::getenv("SSKEYOUT");
  if (keyout_p)
    this->_put_key_prefix = keyout_p;
  else
    this->_put_key_prefix.clear();

  char* keyin_p = std::getenv("SSKEYIN");
  if(keyin_p) {
    char* a = &keyin_p[0];
    char* b = a;
    char parse_char = ';';
    while (*b) {
      if(*b==parse_char) {
	if(a!=b)
	  this->_get_key_prefixes.push_back(std::string(a, b-a));
	a=++b;
      }
      else
	b++;
    }
    if(a!=b)
      this->_get_key_prefixes.push_back(std::string(a, b-a));
  }

  if (this->_get_key_prefixes.size() > 0)
    this->set_data_source(this->_get_key_prefixes[0].c_str());
}

void Client::set_data_source(std::string source_id)
{
  /* This function sets the prefix for fetching keys
  from the database
  */
  bool valid_prefix = false;
  int num_prefix = _get_key_prefixes.size();
  int i = 0;
  for (i=0; i<num_prefix; i++) {
    if (this->_get_key_prefixes[i].compare(source_id)==0) {
      valid_prefix = true;
      break;
    }
  }

  if (valid_prefix)
    this->_get_key_prefix = this->_get_key_prefixes[i];
  else
    throw std::runtime_error("Client error: data source " +
			     std::string(source_id) +
			     "could not be found during client "+
			     "initialization.");
}

inline std::string Client::_put_prefix()
{
  std::string prefix;
  if(this->_put_key_prefix.size()>0)
    prefix =  this->_put_key_prefix + ".";
  return prefix;

}

inline std::string Client::_get_prefix()
{
  std::string prefix;
  if(this->_get_key_prefix.size()>0)
    prefix =  this->_get_key_prefix + ".";
  return prefix;

}

void Client::_append_with_get_prefix(
                     std::vector<std::string>& keys)
{
  /* This function will append each key with the
  get prefix.
  */
  std::vector<std::string>::iterator prefix_it;
  std::vector<std::string>::iterator prefix_it_end;
  prefix_it = keys.begin();
  prefix_it_end = keys.end();
  while(prefix_it != prefix_it_end) {
    *prefix_it = this->_get_prefix() + *prefix_it;
    prefix_it++;
  }
  return;
}

void Client::_append_with_put_prefix(
                     std::vector<std::string>& keys)
{
  /* This function will append each key with the
  put prefix.
  */
  std::vector<std::string>::iterator prefix_it;
  std::vector<std::string>::iterator prefix_it_end;
  prefix_it = keys.begin();
  prefix_it_end = keys.end();
  while(prefix_it != prefix_it_end) {
    *prefix_it = this->_put_prefix() + *prefix_it;
    prefix_it++;
  }
  return;
}

std::vector<std::string>
Client::_get_tmp_names(std::vector<std::string> names,
                       std::string db_prefix)
{
  /* This function returns a map between the original
  names and temporary names that are needed to make sure
  keys hash to the same hash slot.
  */
  std::vector<std::string> tmp;
  std::vector<std::string>::iterator it = names.begin();
  std::vector<std::string>::iterator it_end = names.end();
  while(it!=it_end) {
    std::string new_key = "{" + db_prefix + "}." +
                          *it + ".TMP";
    tmp.push_back(new_key);
    it++;
  }
  return tmp;
}

void Client::_copy_tensors(std::vector<std::string> src,
                           std::vector<std::string> dest)
{
  /* This function will copy a list of tensors from
  src to destination.
  */
  std::vector<std::string>::const_iterator src_it = src.cbegin();
  std::vector<std::string>::const_iterator src_it_end = src.cend();

  std::vector<std::string>::const_iterator dest_it = dest.cbegin();
  std::vector<std::string>::const_iterator dest_it_end = dest.cend();

  while(src_it!=src_it_end && dest_it!=dest_it_end)
  {
    this->copy_tensor(*src_it, *dest_it);
    src_it++;
    dest_it++;
  }
  return;
}

void Client::_delete_keys(std::vector<std::string> keys,
                          std::string db_prefix)
{
  /* This function will delete a list of tensor names in a single
  command.  This assumes they are all in the same hash slot.
  */
  CommandReply reply;
  Command del_cmd;
  del_cmd.add_field("DEL");
  std::vector<std::string>::const_iterator it = keys.cbegin();
  std::vector<std::string>::const_iterator it_end = keys.cend();
  while(it!=it_end) {
    del_cmd.add_field(*it);
    it++;
  }
  reply = this->_execute_command(del_cmd, db_prefix);
  return;
}