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

#include <string_view>
#include "dataset.h"

using namespace SmartRedis;

// DataSet constructor
DataSet::DataSet(const std::string& _name)
 : name(_name)
{
    // NOP
}

// Add a tensor to the DataSet.
void DataSet::add_tensor(const std::string& name,
                         void* data,
                         const std::vector<size_t>& dims,
                         const TensorType type,
                         MemoryLayout mem_layout)
{
    _add_to_tensorpack(name, data, dims,
                             type, mem_layout);
    _metadata.add_string(".tensor_names", name);
}

// Add metadata scalar field (non-string) with value to the DataSet.
// If the field does not exist, it will be created. If the field exists,
// the value will be appended to existing field.
void DataSet::add_meta_scalar(const std::string& name,
                              const void* data,
                              const MetaDataType type)
{
    _metadata.add_scalar(name, data, type);
}

// Add metadata string field with value to the DataSet. If the field
// does not exist, it will be created. If the field exists the value
// will be appended to existing field.
void DataSet::add_meta_string(const std::string& name,
                              const std::string& data)
{
    _metadata.add_string(name, data);
}

// Get the tensor data, dimensions, and type for the tensor in the DataSet.
// This function will allocate and retain management of the memory for
// the tensor data.
void DataSet::get_tensor(const std::string& name,
                         void*& data,
                         std::vector<size_t>& dims,
                         TensorType& type,
                         MemoryLayout mem_layout)
{
    _enforce_tensor_exists(name);
    // Clone the tensor in the DataSet
    TensorBase* tensor = _get_tensorbase_obj(name);
    if (tensor == NULL) {
        throw std::runtime_error("tensor creation failed");
    }
    _tensor_memory.add_tensor(tensor);
    type = tensor->type();
    data = tensor->data_view(mem_layout);
    dims = tensor->dims();
}

// Get the tensor data, dimensions, and type for the tensor in the DataSet.
// This function will allocate and retain management of the memory for the
// tensor data. This is a c-style interface for the tensor dimensions.
void DataSet::get_tensor(const std::string&  name,
                         void*& data,
                         size_t*& dims,
                         size_t& n_dims,
                         TensorType& type,
                         MemoryLayout mem_layout)
{
    std::vector<size_t> dims_vec;
    get_tensor(name, data, dims_vec, type, mem_layout);

    // Get number of dimensions
    dims = _dim_queries.allocate(dims_vec.size());
    n_dims = dims_vec.size();

    // Get size of each dimension
    std::vector<size_t>::const_iterator it = dims_vec.cbegin();
    for (size_t i = 0; it != dims_vec.cend(); it++, i++) {
        dims[i] = *it;
    }
}

// Get tensor data and fill an already allocated array memory space that
// has the specified MemoryLayout. The provided type and dimensions are
// checked against retrieved values to ensure the provided memory space
// is sufficient. This method is the most memory efficient way to retrieve
// tensor data from a DataSet
void DataSet::unpack_tensor(const std::string& name,
                            void* data,
                            const std::vector<size_t>& dims,
                            const TensorType type,
                            MemoryLayout mem_layout)
{
    _enforce_tensor_exists(name);
    _tensorpack.get_tensor(name)->fill_mem_space(data, dims, mem_layout);
}

// Get the metadata scalar field values from the DataSet. The data pointer
// reference will be pointed to newly allocated memory that will contain all
// values in the metadata field. The length variable will be set to the number
// of entries in the allocated memory space to allow for iteration over the values.
// The TensorType enum will be set to the type of the MetaData field.
void DataSet::get_meta_scalars(const std::string& name,
                               void*& data,
                               size_t& length,
                               MetaDataType& type)
{
    _metadata.get_scalar_values(name, data, length, type);
}

// Get the metadata scalar field values from the DataSet. The data pointer
// reference will be pointed to newly allocated memory that will contain all
// values in the metadata field. The length variable will be set to the number
// of entries in the allocated memory space to allow for iteration over the values.
// The TensorType enum will be set to the type of the MetaData field.
void DataSet::get_meta_strings(const std::string& name,
                               char**& data,
                               size_t& n_strings,
                               size_t*& lengths)
{
    _metadata.get_string_values(name, data, n_strings, lengths);
}

// Check if the DataSet has a field
bool DataSet::has_field(const std::string& field_name)
{
    return _metadata.has_field(field_name);
}

// Clear all entries in a DataSet field.
void DataSet::clear_field(const std::string& field_name)
{
    _metadata.clear_field(field_name);
}

// Retrieve the names of the tensors in the DataSet
std::vector<std::string> DataSet::get_tensor_names()
{
    if (_metadata.has_field(".tensor_names"))
        return _metadata.get_string_values(".tensor_names");
    else
        return std::vector<std::string>();

}

// Get the strings in a metadata string field. Because standard C++
// containers are used, memory management is handled by the returned
// std::vectorstd::string.
std::vector<std::string> DataSet::get_meta_strings(const std::string& name)
{
    return _metadata.get_string_values(name);
}

// Get the Tensor type of the Tensor
TensorType DataSet::get_tensor_type(const std::string& name)
{
    return _tensorpack.get_tensor(name)->type();
}

// Add a Tensor (not yet allocated) to the TensorPack
void DataSet::_add_to_tensorpack(const std::string& name,
                                 void* data,
                                 const std::vector<size_t>& dims,
                                 const TensorType type,
                                 const MemoryLayout mem_layout)
{
    _tensorpack.add_tensor(name, data, dims, type, mem_layout);
}

// Retrieve an iterator pointing to the first Tensor in the DataSet
DataSet::tensor_iterator DataSet::tensor_begin()
{
    return _tensorpack.tensor_begin();
}

// Retrieve a const iterator pointing to the first Tensor in the DataSet
DataSet::const_tensor_iterator DataSet::tensor_cbegin()
{
    return _tensorpack.tensor_cbegin();
}

// Retrieve an iterator pointing to the last Tensor in the DataSet
DataSet::tensor_iterator DataSet::tensor_end()
{
    return _tensorpack.tensor_end();
}

// Retrieve a const iterator pointing to the last Tensor in the DataSet
DataSet::const_tensor_iterator DataSet::tensor_cend()
{
    return _tensorpack.tensor_cend();
}

// Returns a vector of std::pair with the field name and the field
// serialization for all fields in the MetaData set.
std::vector<std::pair<std::string, std::string>>
    DataSet::get_metadata_serialization_map()
{
   return _metadata.get_metadata_serialization_map();
}

// Add a serialized field to the DataSet
void DataSet::_add_serialized_field(const std::string& name,
                                   char* buf,
                                   size_t buf_size)
{
    _metadata.add_serialized_field(name, buf, buf_size);
}

// Check and enforce that a tensor must exist or throw an error.
inline void DataSet::_enforce_tensor_exists(const std::string& name)
{
    if (!_tensorpack.tensor_exists(name)) {
        throw std::runtime_error("The tensor " + std::string(name) +
                                 " does not exist in " + name +
                                 " dataset.");
    }
}

// Retrieve the tensor from the DataSet and return a TensorBase object that
// can be used to return tensor information to the user. The returned TensorBase
// object has been dynamically allocated, but not yet tracked for memory
// management in any object.
TensorBase* DataSet::_get_tensorbase_obj(const std::string& name)
{
    return _tensorpack.get_tensor(name)->clone();
}

// EOF
