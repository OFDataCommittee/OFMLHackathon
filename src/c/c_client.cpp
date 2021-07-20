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

#include "c_client.h"

using namespace SmartRedis;

extern "C"
void* SmartRedisCClient(bool cluster)
{
  // Return a pointer to a new Client.
  // The user is responsible for deleting the client via DeleteCClient().
  Client *s = new Client(cluster);
  return reinterpret_cast<void *>(s);
}

extern "C"
void DeleteCClient(void* c_client)
{
  // This function frees the memory associated with the c client.
  Client *s = reinterpret_cast<Client *>(c_client);
  if (NULL != s)
    delete s;
}

extern "C"
void put_dataset(void* c_client, const void* dataset)
{
  // Put a dataset into the database.
  Client *s = reinterpret_cast<Client *>(c_client);
  DataSet* d = reinterpret_cast<DataSet *>(const_cast<void *>(dataset));
  s->put_dataset(*d);
}

extern "C"
void* get_dataset(void* c_client, const char* name, const size_t name_length)
{
  // Return a pointer to a new dataset.  The user is responsible for deleting
  // the dataset via delete_dataset().
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string dataset_name = std::string(name, name_length);
  DataSet *dataset = new DataSet(s->get_dataset(dataset_name));
  return (void*)dataset;
}

extern "C"
void rename_dataset(void* c_client, const char* name,
                    const size_t name_length, const char* new_name,
                    const size_t new_name_length)
{
  // Rename a dataset in the database.
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string name_str = std::string(name, name_length);
  std::string new_name_str = std::string(new_name, new_name_length);
  s->rename_dataset(name_str, new_name_str);
}

extern "C"
void copy_dataset(void* c_client, const char* src_name,
                  const size_t src_name_length, const char* dest_name,
                  const size_t dest_name_length
                  )
{
  // Copy a dataset from teh src_name to the dest_name
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string src_name_str = std::string(src_name, src_name_length);
  std::string dest_name_str = std::string(dest_name, dest_name_length);
  s->copy_dataset(src_name_str, dest_name_str);
}

extern "C"
void delete_dataset(void* c_client, const char* name, const size_t name_length)
{
  // Delete a dataset (all metadata and tensors) from the database.
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string dataset_name = std::string(name, name_length);
  s->delete_dataset(dataset_name);
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
  // Put a tensor of a specified type into the database
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string key_str = std::string(key, key_length);

  std::vector<size_t> dims_vec;
  for (size_t i = 0; i < n_dims; i++)
    dims_vec.push_back(dims[i]);

  s->put_tensor(key_str, data, dims_vec,
                convert_tensor_type(type),
                convert_layout(mem_layout));
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
  // Get a tensor of a specified type from the database
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string key_str = std::string(key, key_length);

  TensorType t_type;
  s->get_tensor(key_str, *result, *dims, *n_dims,
                t_type, convert_layout(mem_layout));

  *type = convert_tensor_type(t_type);
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
  // Get a tensor of a specified type from the database and put the values
  // into the user provided memory space.
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string key_str = std::string(key, key_length);

  std::vector<size_t> dims_vec;
  for (size_t i = 0; i < n_dims; i++)
    dims_vec.push_back(dims[i]);

  s->unpack_tensor(key_str, result, dims_vec,
                   convert_tensor_type(type),
                   convert_layout(mem_layout));
}

extern "C"
void rename_tensor(void* c_client, const char* key,
                   const size_t key_length, const char* new_key,
                   const size_t new_key_length)
{
  // This function renames a tensor from key to new_key
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string key_str = std::string(key, key_length);
  std::string new_key_str = std::string(new_key, new_key_length);
  s->rename_tensor(key_str, new_key_str);
}

extern "C"
void delete_tensor(void* c_client, const char* key, const size_t key_length)
{
  // This function deletes a tensor from the database.
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string key_str = std::string(key, key_length);
  s->delete_tensor(key_str);
}

extern "C"
void copy_tensor(void* c_client, const char* src_name,
                 const size_t src_name_length,
                 const char* dest_name,
                 const size_t dest_name_length)
{
  // This function copies a tensor from src_name to dest_name.
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string src_str = std::string(src_name, src_name_length);
  std::string dest_str = std::string(dest_name, dest_name_length);
  s->copy_tensor(src_str, dest_str);
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
  // This function sets a model stored in a binary file.
  Client *s = reinterpret_cast<Client *>(c_client);
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
  // Set a model stored in a buffer c-string.
  Client *s = reinterpret_cast<Client *>(c_client);
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
}

extern "C"
const char* get_model(void* c_client, const char* key,
               const size_t key_length, size_t* model_length)
{
  // Retrieve the model and model length from the database
  Client *s = reinterpret_cast<Client *>(c_client);
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
  // Put a script in the database that is stored in a file.
  Client* s = (Client *)c_client;
  std::string key_str = std::string(key, key_length);
  std::string device_str = std::string(device, device_length);
  std::string script_file_str = std::string(script_file,
                                            script_file_length);
  s->set_script_from_file(key_str, device_str, script_file_str);
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
  // Put a script in the database from a string
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string key_str = std::string(key, key_length);
  std::string device_str = std::string(device, device_length);
  std::string script_str = std::string(script, script_length);
  s->set_script(key_str, device_str, script_str);
}

extern "C"
void get_script(void* c_client,
                const char* key, const size_t key_length,
                const char** script, size_t* script_length)
{
  // Get the script stored in the database
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string key_str = std::string(key, key_length);
  std::string_view script_str_view = s->get_script(key_str);
  (*script) = script_str_view.data();
  (*script_length) = script_str_view.size();
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
  // Run a script function in the database
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string key_str = std::string(key, key_length);
  std::string function_str = std::string(function, function_length);

  std::vector<std::string> input_vec;
  for (size_t i = 0; i < n_inputs; i++)
    input_vec.push_back(std::string(inputs[i], input_lengths[i]));

  std::vector<std::string> output_vec;
  for (size_t i = 0; i < n_outputs; i++)
    output_vec.push_back(std::string(outputs[i], output_lengths[i]));

  s->run_script(key_str, function_str, input_vec, output_vec);
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
  //  Run a model in the database
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string key_str = std::string(key, key_length);

  std::vector<std::string> input_vec;
  for (size_t i = 0; i < n_inputs; i++)
    input_vec.push_back(std::string(inputs[i], input_lengths[i]));

  std::vector<std::string> output_vec;
  for (size_t i = 0; i < n_outputs; i++)
    output_vec.push_back(std::string(outputs[i], output_lengths[i]));

  s->run_model(key_str, input_vec, output_vec);
}

extern "C"
bool key_exists(void* c_client, const char* key, const size_t key_length)
{
  // Check whether a key exists
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string key_str = std::string(key, key_length);
  return s->key_exists(key_str);
}

extern "C"
bool model_exists(void* c_client, const char* name, const size_t name_length)
{
  // Check whether a model exists
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string name_str = std::string(name, name_length);
  return s->model_exists(name_str);
}

extern "C"
bool tensor_exists(void* c_client, const char* name,
                   const size_t name_length)
{
  // Check whether a tensor exists
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string name_str = std::string(name, name_length);
  return s->tensor_exists(name_str);
}

extern "C"
bool poll_key(void* c_client,
              const char* key, const size_t key_length,
              const int poll_frequency_ms, const int num_tries)
{
  // Poll to wait until a key exists
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string key_str = std::string(key, key_length);
  return s->poll_key(key_str, poll_frequency_ms, num_tries);
}

extern "C"
bool poll_tensor(void* c_client,
                 const char* name, const size_t name_length,
                 const int poll_frequency_ms, const int num_tries)
{
  // Poll to wait until a tensor exists
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string name_str = std::string(name, name_length);
  return s->poll_tensor(name_str, poll_frequency_ms, num_tries);
}

extern "C"
bool poll_model(void* c_client,
                 const char* name, const size_t name_length,
                 const int poll_frequency_ms, const int num_tries)
{
  // Poll to wait until a model exists
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string name_str = std::string(name, name_length);
  return s->poll_model(name_str, poll_frequency_ms, num_tries);
}

extern "C"
void use_model_ensemble_prefix(void* c_client, bool use_prefix)
{
  // Control use of a model ensemble prefix
  Client *s = reinterpret_cast<Client *>(c_client);
  s->use_model_ensemble_prefix(use_prefix);
}

extern "C"
void use_tensor_ensemble_prefix(void* c_client, bool use_prefix)
{
  // Control use of a tensor ensemble prefix
  Client *s = reinterpret_cast<Client *>(c_client);
  s->use_tensor_ensemble_prefix(use_prefix);
}

extern "C"
void set_data_source(void* c_client, const char* source_id, const size_t source_id_length)
{
  // Establish a data source
  Client *s = reinterpret_cast<Client *>(c_client);
  std::string source_id_str = std::string(source_id, source_id_length);
  s->set_data_source(source_id_str);
}
