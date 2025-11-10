/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2023, Hewlett Packard Enterprise
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
#include "srexception.h"
#include "srassert.h"

using namespace SmartRedis;

// Decorator to standardize exception handling in C Dataset API methods
template <class T>
auto c_dataset_api(T&& dataset_api_func, const char* name)
{
  // we create a closure below
  auto decorated = [name, dataset_api_func =
    std::forward<T>(dataset_api_func)](auto&&... args)
  {
    SRError result = SRNoError;
    try {
      dataset_api_func(std::forward<decltype(args)>(args)...);
    }
    catch (const Exception& e) {
      SRSetLastError(e);
      result = e.to_error_code();
    }
    catch (...) {
      std::string msg(
          "A non-standard exception was encountered while executing ");
      msg += name;
      SRSetLastError(SRInternalException(msg));
      result = SRInternalError;
    }
    return result;
  };
  return decorated;
}

// Macro to invoke the decorator with a lambda function
#define MAKE_DATASET_API(stuff)\
    c_dataset_api([&] { stuff }, __func__)()


// Create a new DataSet
extern "C" SRError CDataSet(
  const char* name, const size_t name_length, void** new_dataset)
{
  return MAKE_DATASET_API({
    // Sanity check params
    SR_CHECK_PARAMS(name != NULL && new_dataset != NULL);

    std::string name_str(name, name_length);
    try {
      *new_dataset = NULL;
      DataSet* dataset = new DataSet(name_str);
      *new_dataset = reinterpret_cast<void*>(dataset);
    }
    catch (const std::bad_alloc& e) {
      SRSetLastError(SRBadAllocException("dataset allocation"));
    }
  });
}

// Deallocate a DataSet
extern "C" SRError DeallocateDataSet(void** dataset)
{
  return MAKE_DATASET_API({
    // Sanity check params
    SR_CHECK_PARAMS(dataset != NULL);

    DataSet* d = reinterpret_cast<DataSet*>(*dataset);
    delete d;
    *dataset = NULL;
  });
}

// Add a tensor to the dataset
extern "C" SRError add_tensor(
  void* dataset,
  const char* tensor_name, const size_t tensor_name_length,
  void* data,
  const size_t* dims, const size_t n_dims,
  const SRTensorType type,
  const SRMemoryLayout mem_layout)
{
  return MAKE_DATASET_API({
    // Sanity check params
    SR_CHECK_PARAMS(dataset != NULL && dims != NULL);

    DataSet* d = reinterpret_cast<DataSet*>(dataset);
    std::string tensor_name_str = std::string(tensor_name, tensor_name_length);

    std::vector<size_t> dims_vec;
    dims_vec.assign(dims, dims + n_dims);

    d->add_tensor(tensor_name_str, data, dims_vec, type, mem_layout);
  });
}

// Add a meta data value to the named meta data field
extern "C" SRError add_meta_scalar(
  void* dataset,
  const char* name, const size_t name_length,
  const void* data,
  const SRMetaDataType type)
{
  return MAKE_DATASET_API({
    // Sanity check params
    SR_CHECK_PARAMS(dataset != NULL && name != NULL && data != NULL);

    DataSet* d = reinterpret_cast<DataSet*>(dataset);
    std::string name_str(name, name_length);

    d->add_meta_scalar(name_str, data, type);
  });
}

// Add a meta data value to the named meta data field
extern "C" SRError add_meta_string(
  void* dataset,
  const char* name, const size_t name_length,
  const char* data, const size_t data_length)
{
  return MAKE_DATASET_API({
    // Sanity check params
    SR_CHECK_PARAMS(dataset != NULL && name != NULL && data != NULL);

    DataSet* d = reinterpret_cast<DataSet*>(dataset);
    std::string name_str(name, name_length);
    std::string data_str(data, data_length);

    d->add_meta_string(name_str, data_str);
  });
}

// Get a tensor of a specified type from the database
extern "C" SRError get_dataset_tensor(
  void* dataset,
  const char* name, const size_t name_length,
  void** data,
  size_t** dims, size_t* n_dims,
  SRTensorType* type,
  const SRMemoryLayout mem_layout)
{
  return MAKE_DATASET_API({
    // Sanity check params
    SR_CHECK_PARAMS(dataset != NULL && name != NULL && data != NULL &&
                    dims != NULL && n_dims != NULL && type != NULL);

    DataSet* d = (DataSet* )dataset;
    std::string name_str(name, name_length);

    *type = SRTensorTypeInvalid;
    d->get_tensor(name_str, *data, *dims, *n_dims, *type, mem_layout);
  });
}

// Copy the tensor data buffer into the provided memory space (data).
extern "C" SRError unpack_dataset_tensor(
  void* dataset,
  const char* name, const size_t name_length,
  void* data,
  const size_t* dims, const size_t n_dims,
  const SRTensorType type,
  const SRMemoryLayout mem_layout)
{
  return MAKE_DATASET_API({
    // Sanity check params
    SR_CHECK_PARAMS(dataset != NULL && name != NULL && dims != NULL);

    DataSet* d = reinterpret_cast<DataSet*>(dataset);
    std::string name_str(name, name_length);

    std::vector<size_t> dims_vec;
    dims_vec.assign(dims, dims + n_dims);

    d->unpack_tensor(name_str, data, dims_vec, type, mem_layout);
  });
}

// Get a meta data field
extern "C" SRError get_meta_scalars(
  void* dataset,
  const char* name, const size_t name_length,
  size_t* length,
  SRMetaDataType* type,
  void** scalar_data)
{
  return MAKE_DATASET_API({
    // Sanity check params
    SR_CHECK_PARAMS(dataset != NULL && name != NULL && length != NULL &&
                    type != NULL && scalar_data != NULL);

    DataSet* d = reinterpret_cast<DataSet*>(dataset);
    std::string key_str(name, name_length);

    void* data = NULL;
    d->get_meta_scalars(key_str, data, *length, *type);

    *scalar_data = data;
  });
}

// Get a meta data string field
extern "C" SRError get_meta_strings(
  void* dataset,
  const char* name, const size_t name_length,
  char*** data,
  size_t* n_strings,
  size_t** lengths)
{
  return MAKE_DATASET_API({
    // Sanity check params
    SR_CHECK_PARAMS(dataset != NULL && name != NULL && data != NULL &&
                    n_strings != NULL && lengths != NULL);

    DataSet* d = reinterpret_cast<DataSet*>(dataset);
    std::string key_str(name, name_length);
    d->get_meta_strings(key_str, *data, *n_strings, *lengths);
  });
}

// Retrieve the names of tensors in the DataSet
extern "C" SRError get_tensor_names(
  void* dataset, char*** data, size_t* n_strings, size_t** lengths)
{
  return MAKE_DATASET_API({
    // Sanity check params
    SR_CHECK_PARAMS(dataset != NULL && data != NULL &&
                    n_strings != NULL && lengths != NULL);

    DataSet* d = reinterpret_cast<DataSet*>(dataset);
    d->get_tensor_names(*data, *n_strings, *lengths);
  });
}

// Retrieve the data type of a Tensor in the DataSet
extern "C" SRError get_tensor_type(
    void* dataset, const char* name, size_t name_len, SRTensorType* ttype)
{
  return MAKE_DATASET_API({
    // Sanity check params
    SR_CHECK_PARAMS(dataset != NULL && ttype != NULL);

    DataSet* d = reinterpret_cast<DataSet*>(dataset);
    std::string tensor_name(name, name_len);
    SRTensorType result = d->get_tensor_type(tensor_name);
    *ttype = result;
  });
}

// Retrieve the dimensions of a Tensor in the DataSet
extern "C" SRError get_tensor_dims(
    void* dataset,
    const char* name, size_t name_len,
    size_t** dims,
    size_t *ndims)
{
  return MAKE_DATASET_API({
    // Sanity check params
    SR_CHECK_PARAMS(dataset != NULL && dims != NULL && ndims != NULL);

    DataSet* d = reinterpret_cast<DataSet*>(dataset);
    std::string tensor_name(name, name_len);
    auto result = d->get_tensor_dims(tensor_name);
    size_t num_dims = result.size();

    // Make sure they gave us a big enough buffer
    if (*ndims < num_dims) {
      *ndims = num_dims;
      throw SRBadAllocException(
        "Insufficient space in buffer for tensor dimensions");
    }

    // Give them the results
    *ndims = num_dims;
    int i = 0;
    for (auto it = result.cbegin(); it != result.cend(); ++it, ++i) {
      (*dims)[i] = *it;
    }
  });
}

// Retrieve the names of all metadata fields in the DataSet
extern "C" SRError get_metadata_field_names(
  void* dataset, char*** data, size_t* n_strings, size_t** lengths)
{
  return MAKE_DATASET_API({
    // Sanity check params
    SR_CHECK_PARAMS(dataset != NULL && data != NULL &&
                    n_strings != NULL && lengths != NULL);

    DataSet* d = reinterpret_cast<DataSet*>(dataset);
    d->get_metadata_field_names(*data, *n_strings, *lengths);
  });
}

// Retrieve the data type of a metadata field in the DataSet
extern "C" SRError get_metadata_field_type(
    void* dataset, const char* name, size_t name_len, SRMetaDataType* mdtype)
{
  return MAKE_DATASET_API({
    // Sanity check params
    SR_CHECK_PARAMS(dataset != NULL && mdtype != NULL);

    DataSet* d = reinterpret_cast<DataSet*>(dataset);
    std::string mdf_name(name, name_len);
    SRMetaDataType result = d->get_metadata_field_type(mdf_name);
    *mdtype = result;
  });
}

// Retrieve a string representation of the dataset
const char* dataset_to_string(void* dataset)
{
  static std::string result;
  try
  {
    // Sanity check params
    SR_CHECK_PARAMS(dataset != NULL);

    DataSet* d = reinterpret_cast<DataSet*>(dataset);
    result = d->to_string();
  }
  catch (const Exception& e) {
    SRSetLastError(e);
    result = e.what();
  }
  catch (...) {
    result = "A non-standard exception was encountered while executing ";
    result += __func__;
    SRSetLastError(SRInternalException(result));
  }

  return result.c_str();
}
