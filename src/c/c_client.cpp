#include "c_client.h"

using namespace SILC;

extern "C"
void* SmartSimCClient(bool cluster)
{
  /* Return a pointer to a new Client.
  The user is responsible for deleting the client.
  */
  Client* s = new Client(cluster);
  return (void*)s;
}

extern "C"
void DeleteCClient(void* c_client)
{
  /* This function frees the memory associated
  with the c client.
  */
  Client* s = (Client*)c_client;
  delete s;
  return;
}

extern "C"
void put_dataset(void* c_client, const void* dataset)
{
  /* Put a dataset into the database.
  */
  Client* s = (Client *)c_client;
  DataSet* d = (DataSet*)dataset;
  s->put_dataset(*d);
  return;
}

extern "C"
void* get_dataset(void* c_client, const char* name,
                  const size_t name_length)
{
  /* Return a pointer to a new dataset.  The user is
  responsible for deleting the dataset.
  */
  Client* s = (Client *)c_client;
  std::string dataset_name = std::string(name, name_length);
  DataSet* dataset = new DataSet(s->get_dataset(dataset_name));
  return (void*)dataset;
}

extern "C"
void rename_dataset(void* c_client, const char* name,
                    const size_t name_length, const char* new_name,
                    const size_t new_name_length)
{
  /* Rename a dataset in the database.
  */
  Client* s = (Client *)c_client;
  std::string name_str = std::string(name, name_length);
  std::string new_name_str = std::string(new_name, new_name_length);
  s->rename_dataset(name_str, new_name_str);
  return;
}

extern "C"
void copy_dataset(void* c_client, const char* src_name,
                  const size_t src_name_length, const char* dest_name,
                  const size_t dest_name_length
                  )
{
  /* Copy a dataset from teh src_name to the dest_name
  */
  Client* s = (Client *)c_client;
  std::string src_name_str = std::string(src_name, src_name_length);
  std::string dest_name_str = std::string(dest_name, dest_name_length);
  s->copy_dataset(src_name_str, dest_name_str);
  return;
}

extern "C"
void delete_dataset(void* c_client, const char* name,
                    const size_t name_length)
{
  /* Delete a dataset (all metadata and tensors) from the
  database.
  */
  Client* s = (Client *)c_client;
  std::string dataset_name = std::string(name, name_length);
  s->delete_dataset(dataset_name);
  return;
}

extern "C"
void put_tensor(void* c_client,
                const char* key,
                const size_t key_length,
                void* data,
                const size_t* dims,
                const size_t n_dims,
                CTensorType type,
                CMemoryLayout mem_layout)
{
  /* Put a tensor of a specified type into the database
  */
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);

  std::vector<size_t> dims_vec;
  for(size_t i=0; i<n_dims; i++)
    dims_vec.push_back(dims[i]);

  s->put_tensor(key_str, data, dims_vec,
                convert_tensor_type(type),
                convert_layout(mem_layout));
  return;
}

extern "C"
void get_tensor(void* c_client, const char* key,
                const size_t key_length,
                void** result,
                size_t** dims,
                size_t* n_dims,
                CTensorType* type,
                CMemoryLayout mem_layout)
{
  /* Get a tensor of a specified type from the database
  */
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);

  TensorType t_type;
  s->get_tensor(key_str, *result, *dims, *n_dims,
                t_type, convert_layout(mem_layout));

  *type = convert_tensor_type(t_type);
  return;
}

extern "C"
void unpack_tensor(void* c_client,
                   const char* key,
                   const size_t key_length,
                   void* result,
                   const size_t* dims,
                   const size_t n_dims,
                   CTensorType type,
                   CMemoryLayout mem_layout)
{
  /* Get a tensor of a specified type from the database
  and put the values into the user provided memory space.
  */
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);

  std::vector<size_t> dims_vec;
  for(size_t i=0; i<n_dims; i++)
    dims_vec.push_back(dims[i]);

  s->unpack_tensor(key_str, result, dims_vec,
                   convert_tensor_type(type),
                   convert_layout(mem_layout));
  return;
}

extern "C"
void rename_tensor(void* c_client, const char* key,
                   const size_t key_length, const char* new_key,
                   const size_t new_key_length)
{
  /* This function renames a tensor from key to new_key
  */
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);
  std::string new_key_str = std::string(new_key, new_key_length);
  s->rename_tensor(key_str, new_key_str);
  return;
}

extern "C"
void delete_tensor(void* c_client, const char* key,
                   const size_t key_length)
{
  /* This function deletes a tensor from the database.
  */
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);
  s->delete_tensor(key_str);
  return;
}

extern "C"
void copy_tensor(void* c_client, const char* src_name,
                 const size_t src_name_length,
                 const char* dest_name,
                 const size_t dest_name_length)
{
  /* This function copies a tensor from src_name to
  dest_name.
  */
  Client* s = (Client *)c_client;
  std::string src_str = std::string(src_name, src_name_length);
  std::string dest_str = std::string(dest_name, dest_name_length);
  s->copy_tensor(src_str, dest_str);
  return;
}

extern "C"
void set_model_from_file(void* c_client,
                         const char* key, const size_t key_length,
                         const char* model_file, const size_t model_file_length,
                         const char* backend, const size_t backend_length,
                         const char* device, const size_t device_length,
                         const int batch_size, const int min_batch_size,
                         const char* tag, const size_t tag_length,
                         const char** inputs, const size_t* input_lengths,
                         const size_t n_inputs,
                         const char** outputs, const size_t* output_lengths,
                         const size_t n_outputs)
{
  /* This function sets a model stored in a binary file.
  */
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);
  std::string model_file_str = std::string(model_file, model_file_length);
  std::string backend_str = std::string(backend, backend_length);
  std::string device_str = std::string(device, device_length);
  std::string tag_str = std::string(tag, tag_length);

  // Catch the case where an empty string was sent (default C++ client behavior)
  std::vector<std::string> input_vec;
  if (n_inputs == 1 && input_lengths[0] == 0) {
    input_vec = std::vector<std::string>();
  }
  else {
    for(size_t i=0; i<n_inputs; i++) {
      input_vec.push_back(std::string(inputs[i], input_lengths[i]));
    }
  }

  std::vector<std::string> output_vec;
  if (n_outputs == 1 && output_lengths[0] == 0) {
    output_vec = std::vector<std::string>();
  }
  else {
    for(size_t i=0; i<n_outputs; i++) {
      output_vec.push_back(std::string(outputs[i], output_lengths[i]));
    }
  }

  s->set_model_from_file(key_str, model_file_str, backend_str, device_str,
                         batch_size, min_batch_size, tag_str, input_vec,
                         output_vec);
  return;
}

extern "C"
void set_model(void* c_client,
               const char* key, const size_t key_length,
               const char* model, const size_t model_length,
               const char* backend, const size_t backend_length,
               const char* device, const size_t device_length,
               const int batch_size, const int min_batch_size,
               const char* tag, const size_t tag_length,
               const char** inputs, const size_t* input_lengths,
               const size_t n_inputs,
               const char** outputs, const size_t* output_lengths,
               const size_t n_outputs)
{
  /* This function sets a model stored in a buffer c-string.
  */
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);
  std::string model_str = std::string(model, model_length);
  std::string backend_str = std::string(backend, backend_length);
  std::string device_str = std::string(device, device_length);
  std::string tag_str = std::string(tag, tag_length);

  // Catch the case where an empty string was sent (default C++ client behavior)
  std::vector<std::string> input_vec;
  if (n_inputs == 1 && input_lengths[0] == 0) {
    input_vec = std::vector<std::string>();
  }
  else {
    for(size_t i=0; i<n_inputs; i++) {
      input_vec.push_back(std::string(inputs[i], input_lengths[i]));
    }
  }

  std::vector<std::string> output_vec;
  if (n_outputs == 1 && output_lengths[0] == 0) {
    output_vec = std::vector<std::string>();
  }
  else {
    for(size_t i=0; i<n_outputs; i++) {
      output_vec.push_back(std::string(outputs[i], output_lengths[i]));
    }
  }

  s->set_model(key_str, model_str, backend_str, device_str,
               batch_size, min_batch_size, tag_str, input_vec,
               output_vec);
  return;
}

extern "C"
const char* get_model(void* c_client, const char* key,
               const size_t key_length, size_t* model_length)
{
  /* This function returns the model and model length
  from the database
  */
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);
  std::string_view model_str_view = s->get_model(key_str);
  const char *model = model_str_view.data();
  (*model_length) = model_str_view.size();
  return model;
}

extern "C"
void set_script_from_file(void* c_client,
                          const char* key,
                          const size_t key_length,
                          const char* device,
                          const size_t device_length,
                          const char* script_file,
                          const size_t script_file_length)
{
  /* This function puts a script in the database
  that is stored in a file.
  */
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);
  std::string device_str = std::string(device, device_length);
  std::string script_file_str = std::string(script_file,
                                            script_file_length);
  s->set_script_from_file(key_str, device_str, script_file_str);
  return;
}

extern "C"
void set_script(void* c_client,
                const char* key,
                const size_t key_length,
                const char* device,
                const size_t device_length,
                const char* script,
                const size_t script_length)
{
  /* This function puts a script in the database
  that is stored in a file.
  */
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);
  std::string device_str = std::string(device, device_length);
  std::string script_str = std::string(script, script_length);
  s->set_script(key_str, device_str, script_str);
  return;
}

extern "C"
void get_script(void* c_client,
                const char* key, const size_t key_length,
                const char** script, size_t* script_length)
{
  /* Get the script stored in the database
  */
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);
  std::string_view script_str_view = s->get_script(key_str);
  (*script) = script_str_view.data();
  (*script_length) = script_str_view.size();
  return;
}

extern "C"
void run_script(void* c_client,
                const char* key,
                const size_t key_length,
                const char* function,
                const size_t function_length,
                const char** inputs,
                const size_t* input_lengths,
                const size_t n_inputs,
                const char** outputs,
                const size_t* output_lengths,
                const size_t n_outputs)
{
  /* This function runs a script function in the database
  */
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);
  std::string function_str = std::string(function, function_length);

  std::vector<std::string> input_vec;
  for(size_t i=0; i<n_inputs; i++)
    input_vec.push_back(std::string(inputs[i], input_lengths[i]));

  std::vector<std::string> output_vec;
  for(size_t i=0; i<n_outputs; i++)
    output_vec.push_back(std::string(outputs[i], output_lengths[i]));

  s->run_script(key_str, function_str, input_vec, output_vec);
  return;
}

extern "C"
void run_model(void* c_client,
               const char* key,
               const size_t key_length,
               const char** inputs,
               const size_t* input_lengths,
               const size_t n_inputs,
               const char** outputs,
               const size_t* output_lengths,
               const size_t n_outputs)
{
  /* This function runs a model in the database
  */
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);

  std::vector<std::string> input_vec;
  for(size_t i=0; i<n_inputs; i++)
    input_vec.push_back(std::string(inputs[i], input_lengths[i]));

  std::vector<std::string> output_vec;
  for(size_t i=0; i<n_outputs; i++)
    output_vec.push_back(std::string(outputs[i], output_lengths[i]));

  s->run_model(key_str, input_vec, output_vec);
  return;
}

extern "C"
bool key_exists(void* c_client, const char* key,
                const size_t key_length)
{
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);
  return s->key_exists(key_str);
}

extern "C"
bool poll_key(void* c_client,
              const char* key, const size_t key_length,
              const int poll_frequency_ms, const int num_tries)
{
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);
  return s->poll_key(key_str, poll_frequency_ms, num_tries);
}