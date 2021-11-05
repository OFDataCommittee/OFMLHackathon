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

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "c_client.h"
#include "srexception.h"
#include "srassert.h"

using namespace SmartRedis;

// Return a pointer to a new Client.
// The caller is responsible for deleting the client via DeleteClient().
extern "C"
SRError SmartRedisCClient(bool cluster, void **new_client)
{
  SRError result = sr_ok;
  try {
    // Sanity check params
    SR_CHECK_PARAMS(new_client != NULL);

    Client* s = new Client(cluster);
    *new_client = reinterpret_cast<void*>(s);
  }
  catch (const std::bad_alloc& e) {
    *new_client = NULL;
    sr_set_last_error(smart_bad_alloc("client allocation"));
    result = sr_badalloc;
  }
  catch (const smart_error& e) {
    *new_client = NULL;
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    *new_client = NULL;
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Free the memory associated with the c client.
extern "C"
SRError DeleteCClient(void** c_client)
{
  SRError result = sr_ok;

  try {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL);

    delete reinterpret_cast<Client*>(*c_client);
    *c_client = NULL;
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Put a dataset into the database.
extern "C"
SRError put_dataset(void* c_client, void* dataset)
{
  SRError result = sr_ok;

  try {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && dataset != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    DataSet* d = reinterpret_cast<DataSet*>(dataset);

    s->put_dataset(*d);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Return a pointer to a new dataset.  The user is
// responsible for deleting the dataset via DeallocateeDataSet()
extern "C"
SRError get_dataset(void* c_client, const char* name,
                    const size_t name_length, void **dataset)
{
  SRError result = sr_ok;

  try {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && name != NULL && dataset != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string dataset_name(name, name_length);
    DataSet* d = NULL;

    try {
      d = new DataSet(s->get_dataset(dataset_name));
      *dataset = reinterpret_cast<void*>(d);
    } catch (const std::bad_alloc& e) {
      *dataset = NULL;
      throw smart_bad_alloc("client allocation");
    }
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Rename a dataset in the database.
extern "C"
SRError rename_dataset(void* c_client, const char* name,
                       const size_t name_length, const char* new_name,
                       const size_t new_name_length)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && name != NULL && new_name != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string name_str(name, name_length);
    std::string new_name_str(new_name, new_name_length);

    s->rename_dataset(name_str, new_name_str);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}


// Copy a dataset from the src_name to the dest_name
extern "C"
SRError copy_dataset(void* c_client, const char* src_name,
                    const size_t src_name_length, const char* dest_name,
                    const size_t dest_name_length)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && src_name != NULL && dest_name != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string src_name_str(src_name, src_name_length);
    std::string dest_name_str(dest_name, dest_name_length);

    s->copy_dataset(src_name_str, dest_name_str);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Delete a dataset (all metadata and tensors) from the database.
extern "C"
SRError delete_dataset(void* c_client, const char* name, const size_t name_length)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && name != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string dataset_name(name, name_length);
    s->delete_dataset(dataset_name);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Put a tensor of a specified type into the database
extern "C"
SRError put_tensor(void* c_client,
                  const char* key,
                  const size_t key_length,
                  void* data,
                  const size_t* dims,
                  const size_t n_dims,
                  const CTensorType type,
                  const CMemoryLayout mem_layout)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && key != NULL && 
                    data != NULL && dims != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string key_str(key, key_length);

    std::vector<size_t> dims_vec;
    dims_vec.assign(dims, dims + n_dims);

    s->put_tensor(key_str, data, dims_vec,
                  convert_tensor_type(type),
                  convert_layout(mem_layout));
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Get a tensor of a specified type from the database
extern "C"
SRError get_tensor(void* c_client,
                  const char* key,
                  const size_t key_length,
                  void** result,
                  size_t** dims,
                  size_t* n_dims,
                  CTensorType* type,
                  const CMemoryLayout mem_layout)
{
  SRError outcome = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && key != NULL && result != NULL &&
                    dims != NULL && n_dims != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string key_str(key, key_length);

    TensorType t_type;
    s->get_tensor(key_str, *result, *dims, *n_dims,
                  t_type, convert_layout(mem_layout));

    *type = convert_tensor_type(t_type);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    outcome = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    outcome = sr_internal;
  }

  return outcome;
}

// Get a tensor of a specified type from the database
// and put the values into the user provided memory space.
extern "C"
SRError unpack_tensor(void* c_client,
                     const char* key,
                     const size_t key_length,
                     void* result,
                     const size_t* dims,
                     const size_t n_dims,
                     const CTensorType type,
                     const CMemoryLayout mem_layout)
{
  SRError outcome = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && key != NULL && result != NULL &&
                    dims != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string key_str(key, key_length);

    std::vector<size_t> dims_vec;
    dims_vec.assign(dims, dims + n_dims);

    s->unpack_tensor(key_str, result, dims_vec,
                    convert_tensor_type(type),
                    convert_layout(mem_layout));
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    outcome = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    outcome = sr_internal;
  }

  return outcome;
}

// Rename a tensor from key to new_key
extern "C"
SRError rename_tensor(void* c_client, const char* key,
                     const size_t key_length, const char* new_key,
                     const size_t new_key_length)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && key != NULL && new_key != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string key_str(key, key_length);
    std::string new_key_str(new_key, new_key_length);

    s->rename_tensor(key_str, new_key_str);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Delete a tensor from the database.
extern "C"
SRError delete_tensor(void* c_client, const char* key,
                      const size_t key_length)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && key != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string key_str(key, key_length);

    s->delete_tensor(key_str);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Copy a tensor from src_name to dest_name.
extern "C"
SRError copy_tensor(void* c_client,
                   const char* src_name,
                   const size_t src_name_length,
                   const char* dest_name,
                   const size_t dest_name_length)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && src_name != NULL && dest_name != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string src_str(src_name, src_name_length);
    std::string dest_str(dest_name, dest_name_length);

    s->copy_tensor(src_str, dest_str);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Set a model stored in a binary file.
extern "C"
SRError set_model_from_file(void* c_client,
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
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && key != NULL && model_file != NULL &&
                    backend != NULL && device != NULL && tag != NULL &&
                    inputs != NULL && input_lengths != NULL &&
                    outputs != NULL && output_lengths != NULL);

    // For the inputs and outputs arrays, a single empty string is ok (this means
    // that the array should be skipped) but if more than one entry is present, the
    // strings must be nonzero length
    if (n_inputs != 1 && input_lengths[0] != 0) {
      for (size_t i = 0; i < n_inputs; i++){
        if (inputs[i] == NULL || input_lengths[i] == 0) {
          throw smart_parameter_error(
            std::string("inputs[") + std::to_string(i) + "] is NULL or empty");
        }
      }
    }
    if (n_outputs != 1 && output_lengths[0] != 0) {
      for (size_t i = 0; i < n_outputs; i++) {
        if (outputs[i] == NULL || output_lengths[i] == 0) {
          throw smart_parameter_error(
            std::string("outputs[") + std::to_string(i) + "] is NULL or empty");
        }
      }
    }

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string key_str(key, key_length);
    std::string model_file_str(model_file, model_file_length);
    std::string backend_str(backend, backend_length);
    std::string device_str(device, device_length);
    std::string tag_str(tag, tag_length);

    // Catch the case where an empty string was sent (default C++ client behavior)
    std::vector<std::string> input_vec;
    if (n_inputs != 1 || input_lengths[0] != 0) {
      for (size_t i = 0; i < n_inputs; i++) {
        input_vec.push_back(std::string(inputs[i], input_lengths[i]));
      }
    }

    std::vector<std::string> output_vec;
    if (n_outputs != 1 || output_lengths[0] != 0) {
      for (size_t i = 0; i < n_outputs; i++) {
        output_vec.push_back(std::string(outputs[i], output_lengths[i]));
      }
    }

    s->set_model_from_file(key_str, model_file_str, backend_str, device_str,
                           batch_size, min_batch_size, tag_str, input_vec,
                           output_vec);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Set a model stored in a buffer c-string.
extern "C"
SRError set_model(void* c_client,
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
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && key != NULL && model != NULL &&
                    backend != NULL && device != NULL && tag != NULL &&
                    inputs != NULL && input_lengths != NULL &&
                    outputs != NULL && output_lengths != NULL);

    // For the inputs and outputs arrays, a single empty string is ok (this means
    // that the array should be skipped) but if more than one entry is present, the
    // strings must be nonzero length
    if (n_inputs != 1 && input_lengths[0] != 0) {
      for (size_t i = 0; i < n_inputs; i++){
        if (inputs[i] == NULL || input_lengths[i] == 0) {
          throw smart_parameter_error(
            std::string("inputs[") + std::to_string(i) + "] is NULL or empty");
        }
      }
    }
    if (n_outputs != 1 && output_lengths[0] != 0) {
      for (size_t i = 0; i < n_outputs; i++) {
        if (outputs[i] == NULL || output_lengths[i] == 0) {
          throw smart_parameter_error(
            std::string("outputs[") + std::to_string(i) + "] is NULL or empty");
        }
      }
    }

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string key_str(key, key_length);
    std::string model_str(model, model_length);
    std::string backend_str(backend, backend_length);
    std::string device_str(device, device_length);
    std::string tag_str(tag, tag_length);

    // Catch the case where an empty string was sent (default C++ client behavior)
    std::vector<std::string> input_vec;
    if (n_inputs != 1 || input_lengths[0] != 0) {
      for (size_t i = 0; i < n_inputs; i++) {
        input_vec.push_back(std::string(inputs[i], input_lengths[i]));
      }
    }

    std::vector<std::string> output_vec;
    if (n_outputs != 1 || output_lengths[0] != 0) {
      for (size_t i = 0; i < n_outputs; i++) {
        output_vec.push_back(std::string(outputs[i], output_lengths[i]));
      }
    }

    s->set_model(key_str, model_str, backend_str, device_str,
                batch_size, min_batch_size, tag_str, input_vec,
                output_vec);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Retrieve the model and model length from the database
extern "C"
SRError get_model(void* c_client,
                  const char* key,
                  const size_t key_length,
                  size_t* model_length,
                  const char** model)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && key != NULL && model_length != NULL &&
                    model != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string key_str(key, key_length);
    std::string_view model_str_view(s->get_model(key_str));

    *model_length = model_str_view.size();
    *model = model_str_view.data();
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Put a script in the database that is stored in a file.
extern "C"
SRError set_script_from_file(void* c_client,
                            const char* key,
                            const size_t key_length,
                            const char* device,
                            const size_t device_length,
                            const char* script_file,
                            const size_t script_file_length)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && key != NULL && device != NULL &&
                    script_file != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string key_str(key, key_length);
    std::string device_str(device, device_length);
    std::string script_file_str(script_file, script_file_length);

    s->set_script_from_file(key_str, device_str, script_file_str);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Put a script in the database that is stored in a string.
extern "C"
SRError set_script(void* c_client,
                  const char* key,
                  const size_t key_length,
                  const char* device,
                  const size_t device_length,
                  const char* script,
                  const size_t script_length)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && key != NULL && device != NULL &&
                    script != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);

    std::string key_str(key, key_length);
    std::string device_str(device, device_length);
    std::string script_str(script, script_length);

    s->set_script(key_str, device_str, script_str);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Retrieve the script stored in the database
extern "C"
SRError get_script(void* c_client,
                  const char* key,
                  const size_t key_length,
                  const char** script,
                  size_t* script_length)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && key != NULL && script != NULL &&
                    script_length != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string key_str(key, key_length);
    std::string_view script_str_view(s->get_script(key_str));

    (*script) = script_str_view.data();
    (*script_length) = script_str_view.size();
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Run  a script function in the database
extern "C"
SRError run_script(void* c_client,
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
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && key != NULL && function != NULL &&
                    inputs != NULL && input_lengths != NULL &&
                    outputs != NULL && output_lengths != NULL);

    // Inputs and outputs are mandatory for run_script
    for (size_t i = 0; i < n_inputs; i++){
      if (inputs[i] == NULL || input_lengths[i] == 0) {
        throw smart_parameter_error(
          std::string("inputs[") + std::to_string(i) + "] is NULL or empty");
      }
    }
    for (size_t i = 0; i < n_outputs; i++) {
      if (outputs[i] == NULL || output_lengths[i] == 0) {
        throw smart_parameter_error(
          std::string("outputs[") + std::to_string(i) + "] is NULL or empty");
      }
    }

    std::string key_str(key, key_length);
    std::string function_str(function, function_length);

    std::vector<std::string> input_vec;
    if (n_inputs != 1 || input_lengths[0] != 0) {
      for (size_t i = 0; i < n_inputs; i++) {
        input_vec.push_back(std::string(inputs[i], input_lengths[i]));
      }
    }

    std::vector<std::string> output_vec;
    if (n_outputs != 1 || output_lengths[0] != 0) {
      for (size_t i = 0; i < n_outputs; i++) {
        output_vec.push_back(std::string(outputs[i], output_lengths[i]));
      }
    }

    Client* s = reinterpret_cast<Client*>(c_client);
    s->run_script(key_str, function_str, input_vec, output_vec);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Run a model in the database
extern "C"
SRError run_model(void* c_client,
                 const char* key,
                 const size_t key_length,
                 const char** inputs,
                 const size_t* input_lengths,
                 const size_t n_inputs,
                 const char** outputs,
                 const size_t* output_lengths,
                 const size_t n_outputs)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && key != NULL &&
                    inputs != NULL && input_lengths != NULL &&
                    outputs != NULL && output_lengths != NULL);

    // Inputs and outputs are mandatory for run_script
    for (size_t i = 0; i < n_inputs; i++){
      if (inputs[i] == NULL || input_lengths[i] == 0) {
        throw smart_parameter_error(
          std::string("inputs[") + std::to_string(i) + "] is NULL or empty");
      }
    }
    for (size_t i = 0; i < n_outputs; i++) {
      if (outputs[i] == NULL || output_lengths[i] == 0) {
        throw smart_parameter_error(
          std::string("outputs[") + std::to_string(i) + "] is NULL or empty");
      }
    }

    std::string key_str(key, key_length);

    std::vector<std::string> input_vec;
    if (n_inputs != 1 || input_lengths[0] != 0) {
      for (size_t i = 0; i < n_inputs; i++) {
        input_vec.push_back(std::string(inputs[i], input_lengths[i]));
      }
    }

    std::vector<std::string> output_vec;
    if (n_outputs != 1 || output_lengths[0] != 0) {
      for (size_t i = 0; i < n_outputs; i++) {
        output_vec.push_back(std::string(outputs[i], output_lengths[i]));
      }
    }

    Client* s = reinterpret_cast<Client*>(c_client);
    s->run_model(key_str, input_vec, output_vec);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Check whether a key exists in the database
extern "C"
SRError key_exists(void* c_client, const char* key, const size_t key_length,
                   bool* exists)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && key != NULL && exists != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string key_str(key, key_length);

    *exists = s->key_exists(key_str);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Check whether a model exists in the database
extern "C"
SRError model_exists(void* c_client, const char* name, const size_t name_length,
                     bool* exists)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && name != NULL && exists != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string name_str(name, name_length);

    *exists = s->model_exists(name_str);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Check whether a tensor exists in the database
extern "C"
SRError tensor_exists(void* c_client, const char* name, const size_t name_length,
                      bool* exists)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && name != NULL && exists != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string name_str(name, name_length);

    *exists = s->tensor_exists(name_str);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Delay until a dataset exists in the database
extern "C"
SRError dataset_exists(void* c_client, const char* name, const size_t name_length,
                       bool* exists)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && name != NULL && exists != NULL);

    Client* s = reinterpret_cast<Client *>(c_client);
    std::string name_str(name, name_length);

    *exists = s->dataset_exists(name_str);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Delay until a key exists in the database
extern "C"
SRError poll_key(void* c_client,
                 const char* key,
                 const size_t key_length,
                 const int poll_frequency_ms,
                 const int num_tries,
                 bool* exists)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && key != NULL && exists != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string key_str(key, key_length);

    *exists = s->poll_key(key_str, poll_frequency_ms, num_tries);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Delay until a model exists in the database
extern "C"
SRError poll_model(void* c_client,
                   const char* name,
                   const size_t name_length,
                   const int poll_frequency_ms,
                   const int num_tries,
                   bool* exists)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && name != NULL && exists != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string name_str(name, name_length);

    *exists = s->poll_model(name_str, poll_frequency_ms, num_tries);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Delay until a tensor exists in the database
extern "C"
SRError poll_tensor(void* c_client,
                 const char* name,
                 const size_t name_length,
                 const int poll_frequency_ms,
                 const int num_tries,
                 bool* exists)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && name != NULL && exists != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string name_str(name, name_length);

    *exists = s->poll_tensor(name_str, poll_frequency_ms, num_tries);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Establish a data source
extern "C"
SRError set_data_source(void* c_client,
                        const char* source_id,
                        const size_t source_id_length)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL && source_id != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    std::string source_id_str(source_id, source_id_length);

    s->set_data_source(source_id_str);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Control whether a model ensemble prefix is used
extern "C"
SRError use_model_ensemble_prefix(void* c_client, bool use_prefix)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    s->use_model_ensemble_prefix(use_prefix);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}

// Control whether a tensor ensemble prefix is used
extern "C"
SRError use_tensor_ensemble_prefix(void* c_client, bool use_prefix)
{
  SRError result = sr_ok;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(c_client != NULL);

    Client* s = reinterpret_cast<Client*>(c_client);
    s->use_tensor_ensemble_prefix(use_prefix);
  }
  catch (const smart_error& e) {
    sr_set_last_error(e);
    result = e.to_error_code();
  }
  catch (...) {
    sr_set_last_error(smart_internal_error("Unknown exception occurred"));
    result = sr_internal;
  }

  return result;
}
