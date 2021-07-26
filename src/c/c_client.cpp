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
#include <iostream>

using namespace SmartRedis;


// Return a pointer to a new Client.
// The caller is responsible for deleting the client via DeleteClient().
extern "C"
void *SmartRedisCClient(bool cluster)
{
  std::cout << "SmartRedisCClient" << std::endl;
	Client *s = NULL;
	try {
		s = new Client(cluster);
	} catch (...) {
		s = NULL;
	}
  return (void *)s;
}

// Free the memory associated with the c client.
extern "C"
void DeleteCClient(void *c_client)
{
  std::cout << "SmartRedisCClientDeleteCClient" << std::endl;
  // Sanity check params
  if (NULL == c_client)
	  return; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client*)c_client;
  delete s;
}

// Put a dataset into the database.
extern "C"
void put_dataset(void *c_client, const void *dataset)
{
  std::cout << "put_dataset" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == dataset)
	  return; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client *)c_client;
  DataSet *d = (DataSet *)dataset;
  s->put_dataset(*d);
  return;
}

// Return a pointer to a new dataset.  The user is
// responsible for deleting the dataset via DeallocateeDataSet()
extern "C"
void *get_dataset(void *c_client, const char *name,
                  const size_t name_length)
{
  std::cout << "get_dataset" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == name)
	  return NULL; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client *)c_client;
  std::string dataset_name(name, name_length);
  DataSet *dataset = NULL;
  try {
	  dataset = new DataSet(s->get_dataset(dataset_name));
  } catch (...) {
	  dataset = NULL;
  }
  return (void*)dataset;
}

// Rename a dataset in the database.
extern "C"
void rename_dataset(void *c_client, const char *name,
                    const size_t name_length, const char *new_name,
                    const size_t new_name_length)
{
  std::cout << "rename_dataset" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == name || NULL == new_name)
	  return; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client *)c_client;
  std::string name_str(name, name_length);
  std::string new_name_str(new_name, new_name_length);
  s->rename_dataset(name_str, new_name_str);
  return;
}

// Copy a dataset from the src_name to the dest_name
extern "C"
void copy_dataset(void *c_client, const char *src_name,
                  const size_t src_name_length, const char *dest_name,
                  const size_t dest_name_length
                  )
{
  std::cout << "copy_dataset" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == src_name || NULL == dest_name)
	  return; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client *)c_client;
  std::string src_name_str(src_name, src_name_length);
  std::string dest_name_str(dest_name, dest_name_length);
  s->copy_dataset(src_name_str, dest_name_str);
  return;
}

// Delete a dataset (all metadata and tensors) from the database.
extern "C"
void delete_dataset(void *c_client, const char *name,
                    const size_t name_length)
{
  std::cout << "delete_dataset" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == name)
	  return; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client *)c_client;
  std::string dataset_name(name, name_length);
  s->delete_dataset(dataset_name);
  return;
}

// Put a tensor of a specified type into the database
extern "C"
void put_tensor(void *c_client,
                const char *key,
                const size_t key_length,
                void *data,
                const size_t *dims,
                const size_t n_dims,
                const CTensorType type,
                const CMemoryLayout mem_layout)
{
  std::cout << "put_tensor" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == key || NULL == data || NULL == dims)
	  return; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client *)c_client;
  std::string key_str(key, key_length);

  std::vector<size_t> dims_vec;
  dims_vec.assign(dims, dims + n_dims);

  s->put_tensor(key_str, data, dims_vec,
                convert_tensor_type(type),
                convert_layout(mem_layout));
  return;
}

// Get a tensor of a specified type from the database
extern "C"
void get_tensor(void *c_client,
				const char *key,
                const size_t key_length,
                void **result,
                size_t **dims,
                size_t *n_dims,
                CTensorType *type,
                const CMemoryLayout mem_layout)
{
  std::cout << "get_tensor" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == key || NULL == result || NULL == dims ||
	  NULL == n_dims)
	  return; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client *)c_client;
  std::string key_str(key, key_length);

  TensorType t_type;
  s->get_tensor(key_str, *result, *dims, *n_dims,
                t_type, convert_layout(mem_layout));

  *type = convert_tensor_type(t_type);
  return;
}

// Get a tensor of a specified type from the database
// and put the values into the user provided memory space.
extern "C"
void unpack_tensor(void *c_client,
                   const char *key,
                   const size_t key_length,
                   void *result,
                   const size_t *dims,
                   const size_t n_dims,
                   const CTensorType type,
                   const CMemoryLayout mem_layout)
{
  std::cout << "unpack_tensor" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == key || NULL == result || NULL == dims)
	  return; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client *)c_client;
  std::string key_str(key, key_length);

  std::vector<size_t> dims_vec;
  dims_vec.assign(dims, dims + n_dims);

  s->unpack_tensor(key_str, result, dims_vec,
                   convert_tensor_type(type),
                   convert_layout(mem_layout));
  return;
}

// Rename a tensor from key to new_key
extern "C"
void rename_tensor(void *c_client, const char *key,
                   const size_t key_length, const char *new_key,
                   const size_t new_key_length)
{
  std::cout << "rename_tensor" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == key || NULL == new_key)
	  return; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client *)c_client;
  std::string key_str(key, key_length);
  std::string new_key_str(new_key, new_key_length);
  s->rename_tensor(key_str, new_key_str);
  return;
}

// Delete a tensor from the database.
extern "C"
void delete_tensor(void *c_client, const char *key,
                   const size_t key_length)
{
  std::cout << "delete_tensor" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == key)
	  return; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client *)c_client;
  std::string key_str(key, key_length);
  s->delete_tensor(key_str);
  return;
}

// Copy a tensor from src_name to dest_name.
extern "C"
void copy_tensor(void *c_client,
				 const char *src_name,
                 const size_t src_name_length,
                 const char *dest_name,
                 const size_t dest_name_length)
{
  std::cout << "copy_tensor" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == src_name || NULL == dest_name)
	  return; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client *)c_client;
  std::string src_str(src_name, src_name_length);
  std::string dest_str(dest_name, dest_name_length);
  s->copy_tensor(src_str, dest_str);
  return;
}

// Set a model stored in a binary file.
extern "C"
void set_model_from_file(void *c_client,
                         const char *key, const size_t key_length,
                         const char *model_file, const size_t model_file_length,
                         const char *backend, const size_t backend_length,
                         const char *device, const size_t device_length,
                         const int batch_size, const int min_batch_size,
                         const char *tag, const size_t tag_length,
                         const char **inputs, const size_t *input_lengths,
                         const size_t n_inputs,
                         const char **outputs, const size_t *output_lengths,
                         const size_t n_outputs)
{
  std::cout << "set_model_from_file" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == key || NULL == model_file || NULL == backend ||
	  NULL == device || NULL == tag || NULL == inputs || NULL == input_lengths ||
	  NULL == outputs || NULL == output_lengths)
	  return; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client *)c_client;
  std::string key_str(key, key_length);
  std::string model_file_str(model_file, model_file_length);
  std::string backend_str(backend, backend_length);
  std::string device_str(device, device_length);
  std::string tag_str(tag, tag_length);

  // Catch the case where an empty string was sent (default C++ client behavior)
  std::vector<std::string> input_vec;
  if (1 != n_inputs || 0 != input_lengths[0]) {
    for(size_t i=0; i<n_inputs; i++) {
	  if (NULL != inputs[i])
		  input_vec.push_back(std::string(inputs[i], input_lengths[i]));
	  else
		  input_vec.push_back(std::string(""));
	}
  }

  std::vector<std::string> output_vec;
  if (1 != n_outputs || 0 != output_lengths[0]) {
    for(size_t i=0; i<n_outputs; i++) {
	  if (NULL != outputs[i])
		  output_vec.push_back(std::string(outputs[i], output_lengths[i]));
	  else
		  output_vec.push_back(std::string(""));
	}
  }

  s->set_model_from_file(key_str, model_file_str, backend_str, device_str,
                         batch_size, min_batch_size, tag_str, input_vec,
                         output_vec);
  return;
}

// Set a model stored in a buffer c-string.
extern "C"
void set_model(void *c_client,
               const char *key, const size_t key_length,
               const char *model, const size_t model_length,
               const char *backend, const size_t backend_length,
               const char *device, const size_t device_length,
               const int batch_size, const int min_batch_size,
               const char *tag, const size_t tag_length,
               const char **inputs, const size_t *input_lengths,
               const size_t n_inputs,
               const char **outputs, const size_t *output_lengths,
               const size_t n_outputs)
{
  std::cout << "set_model" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == key || NULL == model || NULL == backend ||
	  NULL == device || NULL == tag || NULL == inputs || NULL == input_lengths ||
	  NULL == outputs || NULL == output_lengths)
	  return; // Nothing we can do, so just bail. (error reporting to come later)
  
  Client *s = (Client *)c_client;
  std::string key_str(key, key_length);
  std::string model_str(model, model_length);
  std::string backend_str(backend, backend_length);
  std::string device_str(device, device_length);
  std::string tag_str(tag, tag_length);

  // Catch the case where an empty string was sent (default C++ client behavior)
  std::vector<std::string> input_vec;
  if (1 != n_inputs || 0 != input_lengths[0]) {
	for(size_t i=0; i<n_inputs; i++) {
	  if (NULL != inputs[i])
		  input_vec.push_back(std::string(inputs[i], input_lengths[i]));
	  else
		  input_vec.push_back(std::string(""));
	}
  }

  std::vector<std::string> output_vec;
  if (1 != n_outputs || 0 != output_lengths[0]) {
	for(size_t i=0; i<n_outputs; i++) {
	  if (NULL != outputs[i])
		  output_vec.push_back(std::string(outputs[i], output_lengths[i]));
	  else
		  output_vec.push_back(std::string(""));
	}
  }

  s->set_model(key_str, model_str, backend_str, device_str,
               batch_size, min_batch_size, tag_str, input_vec,
               output_vec);
  return;
}

// Retrieve the model and model length from the database
extern "C"
const char *get_model(void *c_client,
					  const char *key,
					  const size_t key_length,
					  size_t *model_length)
{
  std::cout << "get_model" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == key || NULL == model_length)
	  return NULL; // Nothing we can do, so just bail. (error reporting to come later)
  
  Client *s = (Client *)c_client;
  std::string key_str(key, key_length);
  std::string_view model_str_view(s->get_model(key_str));
  const char *model = model_str_view.data();
  (*model_length) = model_str_view.size();
  return model;
}

// Put a script in the database that is stored in a file.
extern "C"
void set_script_from_file(void *c_client,
                          const char *key,
                          const size_t key_length,
                          const char *device,
                          const size_t device_length,
                          const char *script_file,
                          const size_t script_file_length)
{
  std::cout << "set_script_from_file" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == key || NULL == device || NULL == script_file)
	  return; // Nothing we can do, so just bail. (error reporting to come later)
  
  Client *s = (Client *)c_client;
  std::string key_str(key, key_length);
  std::string device_str(device, device_length);
  std::string script_file_str(script_file, script_file_length);
  s->set_script_from_file(key_str, device_str, script_file_str);
  return;
}

// Put a script in the database that is stored in a string.
extern "C"
void set_script(void *c_client,
                const char *key,
                const size_t key_length,
                const char *device,
                const size_t device_length,
                const char *script,
                const size_t script_length)
{
  std::cout << "set_script" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == key || NULL == device || NULL == script)
	  return; // Nothing we can do, so just bail. (error reporting to come later)
  
  Client *s = (Client *)c_client;
  
  std::string key_str(key, key_length);
  std::string device_str(device, device_length);
  std::string script_str(script, script_length);
  s->set_script(key_str, device_str, script_str);
  return;
}

// Retrieve the script stored in the database
extern "C"
void get_script(void *c_client,
                const char *key,
				const size_t key_length,
                const char **script,
				size_t *script_length)
{
  std::cout << "get_script" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == key || NULL == script || NULL == script_length)
	  return; // Nothing we can do, so just bail. (error reporting to come later)
  
  Client *s = (Client *)c_client;
  std::string key_str(key, key_length);
  std::string_view script_str_view(s->get_script(key_str));
  (*script) = script_str_view.data();
  (*script_length) = script_str_view.size();
  return;
}

// Run  a script function in the database
extern "C"
void run_script(void *c_client,
                const char *key,
                const size_t key_length,
                const char *function,
                const size_t function_length,
                const char **inputs,
                const size_t *input_lengths,
                const size_t n_inputs,
                const char **outputs,
                const size_t *output_lengths,
                const size_t n_outputs)
{
  std::cout << "run_script" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == key || NULL == function || NULL == inputs ||
	  NULL == input_lengths || NULL == outputs || NULL == output_lengths) {
	  return; // Nothing we can do, so just bail. (error reporting to come later)
  }

  std::string key_str(key, key_length);
  std::string function_str(function, function_length);

  std::vector<std::string> input_vec;
  if (1 != n_inputs || 0 != input_lengths[0]) {
    for(size_t i=0; i<n_inputs; i++) {
	  if (NULL != inputs[i])
		  input_vec.push_back(std::string(inputs[i], input_lengths[i]));
	  else
		  input_vec.push_back(std::string(""));
	}
  }

  std::vector<std::string> output_vec;
  if (1 != n_outputs || 0 != output_lengths[0]) {
    for(size_t i=0; i<n_outputs; i++) {
	  if (NULL != outputs[i])
		  output_vec.push_back(std::string(outputs[i], output_lengths[i]));
	  else
		  output_vec.push_back(std::string(""));
	}
  }

  Client *s = (Client *)c_client;
  s->run_script(key_str, function_str, input_vec, output_vec);
  return;
}

// Run a model in the database
extern "C"
void run_model(void *c_client,
               const char *key,
               const size_t key_length,
               const char **inputs,
               const size_t *input_lengths,
               const size_t n_inputs,
               const char **outputs,
               const size_t *output_lengths,
               const size_t n_outputs)
{
  std::cout << "run_model" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == key || NULL == inputs || NULL == input_lengths ||
	  NULL == outputs || NULL == output_lengths) {
	  return; // Nothing we can do, so just bail. (error reporting to come later)
  }
  
  std::string key_str(key, key_length);

  std::vector<std::string> input_vec;
  if (1 != n_inputs || 0 != input_lengths[0]) {
    for (size_t i = 0; i < n_inputs; i++) {
	  if (NULL != inputs[i])
		  input_vec.push_back(std::string(inputs[i], input_lengths[i]));
	  else
		  input_vec.push_back(std::string(""));
	}
  }

  std::vector<std::string> output_vec;
  if (1 != n_outputs || 0 != output_lengths[0]) {
    for (size_t i = 0; i < n_outputs; i++) {
	  if (NULL != outputs[i])
		  output_vec.push_back(std::string(outputs[i], output_lengths[i]));
	  else
		  output_vec.push_back(std::string(""));
	}
  }

  Client *s = (Client *)c_client;
  s->run_model(key_str, input_vec, output_vec);
  return;
}

// Check whether a key exists in the database
extern "C"
bool key_exists(void *c_client, const char *key, const size_t key_length)
{
  std::cout << "key_exists" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == key)
	  return false;

  Client *s = (Client *)c_client;
  std::string key_str(key, key_length);
  return s->key_exists(key_str);
}

// Check whether a model exists in the database
extern "C"
bool model_exists(void *c_client, const char *name, const size_t name_length)
{
  std::cout << "model_exists" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == name)
	  return false;

  Client *s = (Client *)c_client;
  std::string name_str(name, name_length);
  return s->model_exists(name_str);
}

// Check whether a tensor exists in the database
extern "C"
bool tensor_exists(void *c_client, const char *name, const size_t name_length)
{
  std::cout << "tensor_exists" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == name)
	  return false;

  Client *s = (Client *)c_client;
  std::string name_str(name, name_length);
  return s->tensor_exists(name_str);
}

// Delay until a key exists in the database
extern "C"
bool poll_key(void *c_client,
              const char *key,
			  const size_t key_length,
              const int poll_frequency_ms,
			  const int num_tries)
{
  std::cout << "poll_key" << std::endl;
  // Sanity check params
  if (NULL == c_client)
	  return false;

  Client *s = (Client *)c_client;
  std::string key_str(key, key_length);
  return s->poll_key(key_str, poll_frequency_ms, num_tries);
}

// Delay until a model exists in the database
extern "C"
bool poll_model(void *c_client,
				const char *name,
				const size_t name_length,
				const int poll_frequency_ms,
				const int num_tries)
{
  std::cout << "poll_model" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == name)
	  return false;

  Client *s = (Client *)c_client;
  std::string name_str(name, name_length);
  return s->poll_model(name_str, poll_frequency_ms, num_tries);
}

// Delay until a tensor exists in the database
extern "C"
bool poll_tensor(void *c_client,
                 const char *name,
				 const size_t name_length,
                 const int poll_frequency_ms,
				 const int num_tries)
{
  std::cout << "poll_tensor" << std::endl;
  // Sanity check params
  if (NULL == c_client || NULL == name)
	  return false;

  Client *s = (Client *)c_client;
  std::string name_str(name, name_length);
  return s->poll_tensor(name_str, poll_frequency_ms, num_tries);
}

// Establish a data source
extern "C"
void set_data_source(void *c_client,
                     const char *source_id,
					 const size_t source_id_length)
{
  std::cout << "set_data_source" << std::endl;
  // Sanity check params
  if (NULL == c_client)
	  return; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client *)c_client;
  std::string source_id_str(source_id, source_id_length);
  s->set_data_source(source_id_str);
  return;
}

// Control whether a model ensemble prefix is used
extern "C"
void use_model_ensemble_prefix(void *c_client, bool use_prefix)
{
  std::cout << "use_model_ensemble_prefix" << std::endl;
  // Sanity check params
  if (NULL == c_client)
	  return; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client *)c_client;
  s->use_model_ensemble_prefix(use_prefix);
  return;
}

// Control whether a tensor ensemble prefix is used
extern "C"
void use_tensor_ensemble_prefix(void *c_client, bool use_prefix)
{
  std::cout << "use_tensor_ensemble_prefix" << std::endl;
  // Sanity check params
  if (NULL == c_client)
	  return; // Nothing we can do, so just bail. (error reporting to come later)

  Client *s = (Client *)c_client;
  s->use_tensor_ensemble_prefix(use_prefix);
  return;
}
