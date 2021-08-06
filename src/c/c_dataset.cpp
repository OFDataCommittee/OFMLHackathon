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

#include "c_dataset.h"

using namespace SmartRedis;

// Create a new DataSet.
// The user is responsible for deallocating the DataSet via DeallocateeDataSet()
extern "C"
void* CDataSet(const char* name, const size_t name_length)
{
  // Sanity check parameters
  if (name == NULL)
    return NULL;

  std::string name_str(name, name_length);

  DataSet* dataset = NULL;
  try {
    dataset = new DataSet(name_str);
  } catch (...) {
    dataset = NULL;
  }

  return reinterpret_cast<void* >(dataset);
}

// Deallocate a DataSet
extern "C"
void DeallocateeDataSet(void* dataset)
{
  // Sanity check parameters
  if (dataset == NULL)
    return;

  DataSet* d = reinterpret_cast<DataSet*>(dataset);
  delete d;
}

// Add a tensor to the dataset.
extern "C"
void add_tensor(void* dataset,
                const char* tensor_name,
                const size_t tensor_name_length,
                void* data,
                const size_t* dims,
                const size_t n_dims,
                const CTensorType type,
                const CMemoryLayout mem_layout)
{
  // Sanity check parameters
  DataSet* d = reinterpret_cast<DataSet*>(dataset);
  std::string tensor_name_str = std::string(tensor_name, tensor_name_length);

  std::vector<size_t> dims_vec;
  dims_vec.assign(dims, dims + n_dims);

  d->add_tensor(tensor_name_str, data, dims_vec,
                convert_tensor_type(type),
                convert_layout(mem_layout));
}

// Add a meta data value to the named meta data field
extern "C"
void add_meta_scalar(void* dataset,
                     const char* name,
                     const size_t name_length,
                     const void* data,
                     const CMetaDataType type)
{
  // Sanity check parameters
  if (dataset == NULL || name == NULL || data == NULL)
    return;

  DataSet* d = reinterpret_cast<DataSet*>(dataset);
  std::string name_str(name, name_length);

  d->add_meta_scalar(name_str, data,
                     convert_metadata_type(type));
}

// Add a meta data value to the named meta data field
extern "C"
void add_meta_string(void* dataset,
                     const char* name,
                     const size_t name_length,
                     const char* data,
                     const size_t data_length)
{
  // Sanity check parameters
  if (dataset == NULL || name == NULL || data == NULL)
    return;

  DataSet* d = reinterpret_cast<DataSet*>(dataset);
  std::string name_str(name, name_length);
  std::string data_str(data, data_length);

  d->add_meta_string(name_str, data_str);
}

// Get a tensor of a specified type from the database.
// This function may allocate new memory for the tensor.
// This memory will be deleted when the user deletes the
// DataSet object.
extern "C"
void get_dataset_tensor(void* dataset,
                        const char* name,
                        const size_t name_length,
                        void** data,
                        size_t** dims,
                        size_t* n_dims,
                        CTensorType* type,
                        const CMemoryLayout mem_layout)
{
  // Sanity check parameters
  if (dataset == NULL || name == NULL || data == NULL ||
      dims == NULL || n_dims == NULL || type == NULL)
    return;

  DataSet* d = (DataSet* )dataset;
  std::string name_str(name, name_length);

  TensorType t_type = TensorType::undefined;
  d->get_tensor(name_str,* data,* dims,* n_dims,
                t_type, convert_layout(mem_layout));
  *type = convert_tensor_type(t_type);
}

// Copy the tensor data buffer into the provided memory space (data).
extern "C"
void unpack_dataset_tensor(void* dataset,
                           const char* name,
                           const size_t name_length,
                           void* data,
                           const size_t* dims,
                           const size_t n_dims,
                           const CTensorType type,
                           const CMemoryLayout mem_layout)
{
  // Sanity check parameters
  if (dataset == NULL || name == NULL || data == NULL || dims == NULL)
    return;

  DataSet* d = reinterpret_cast<DataSet*>(dataset);
  std::string name_str(name, name_length);

  std::vector<size_t> dims_vec;
  dims_vec.assign(dims, dims + n_dims);

  d->unpack_tensor(name_str, data, dims_vec,
                   convert_tensor_type(type),
                   convert_layout(mem_layout));
}

// Get a meta data field. This method may allocate
// memory that is cleared when the user deletes the
// DataSet object
extern "C"
void* get_meta_scalars(void* dataset,
                       const char* name,
                       const size_t name_length,
                       size_t* length,
                       CMetaDataType* type)
{
  // Sanity check parameters
  if (dataset == NULL || name == NULL || length == NULL || type == NULL)
    return NULL;

  DataSet* d = reinterpret_cast<DataSet*>(dataset);
  std::string key_str(name, name_length);
  void* data = NULL;
  MetaDataType m_type;

  d->get_meta_scalars(key_str, data, *length, m_type);

  *type = convert_metadata_type(m_type);
  return data;
}

// Get a meta data string field. This method may
// allocate memory that is cleared when the user
// deletes the DataSet object.
extern "C"
void get_meta_strings(void* dataset,
                      const char* name,
                      const size_t name_length,
                      char*** data,
                      size_t* n_strings,
                      size_t** lengths)
{
  // Sanity check parameters
  if (dataset == NULL || name == NULL || data == NULL ||
      n_strings == NULL || lengths == NULL)
    return;

  DataSet* d = reinterpret_cast<DataSet*>(dataset);
  std::string key_str(name, name_length);
  d->get_meta_strings(key_str, *data, *n_strings, *lengths);
}
