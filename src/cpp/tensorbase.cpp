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
#include "tensorbase.h"
#include "srexception.h"

using namespace SmartRedis;

// TensorBase constructor
TensorBase::TensorBase(const std::string& name,
                       void* data,
                       const std::vector<size_t>& dims,
                       const SRTensorType type,
                       const SRMemoryLayout mem_layout)
{
    /* The TensorBase constructor makes a copy of the
    name, type, and dims associated with the tensor.
    The provided data is copied into a memory space
    owned by the tensor.
    */

    _check_inputs(data, name, dims);
    _name = name;
    _type = type;
    _dims = dims;
}

// TensorBase copy constructor
TensorBase::TensorBase(const TensorBase& tb)
{
    // check for self-assignment
    if (&tb == this)
        return;

    // deep copy of tensor data
    _dims = std::vector<size_t>(tb._dims);
    _name = std::string(tb._name);
    _type = SRTensorType(tb._type);
}

// TensorBase move constructor
TensorBase::TensorBase(TensorBase&& tb)
{
    // check for self-assignment
    if (&tb == this)
        return;

    // Move data
    _name = std::move(tb._name);
    _type = std::move(tb._type);
    _dims = std::move(tb._dims);
    _data = tb._data;

    // Mark that the data is no longer owned by the source
    tb._data = NULL;
}

// TensorBase destructor
TensorBase::~TensorBase()
{
    if (_data != NULL) {
        delete(reinterpret_cast<unsigned char *>(_data));
        _data = NULL;
    }
}

// TensorBase copy assignment operator
TensorBase& TensorBase::operator=(const TensorBase& tb)
{
    // check for self-assignment
    if (&tb == this)
        return *this;

    // deep copy tensor data
    _name = tb._name;
    _type = tb._type;
    _dims = tb._dims;

    // Erase our old data
    if (_data != NULL) {
        delete(reinterpret_cast<unsigned char *>(_data));
        _data = NULL;
    }

    // NOTE: The actual tensor data will be copied by the child class
    // (template) after it calls this assignment operator.
    // This routine should never be called directly

    // Done
   return *this;
}

// TensorBase move assignment operator
TensorBase& TensorBase::operator=(TensorBase&& tb)
{
    // check for self-assignment
    if (&tb == this)
        return *this;

    // Move tensor data
    _name = std::move(tb._name);
    _type = std::move(tb._type);
    _dims = std::move(tb._dims);

    // Erase our old data and assume ownership of tb's data
    if (_data != NULL)
        delete(reinterpret_cast<unsigned char *>(_data));
    _data = tb._data;
    tb._data = NULL;

    // Done
    return *this;
}

// Retrieve the tensor name.
std::string TensorBase::name()
{
    return _name;
}

// Retrieve the tensor type.
SRTensorType TensorBase::type()
{
   return _type;
}

// Retrieve the string version of the tensor type.
std::string TensorBase::type_str()
{
    return TENSOR_STR_MAP.at(type());
}

// Retrieve the tensor dims.
std::vector<size_t> TensorBase::dims()
{
   return _dims;
}

// Retrieve the total number of values in the tensor.
size_t TensorBase::num_values()
{
    if (_dims.size() == 0)
        throw SRRuntimeException("Invalid dimensionality for tensor detected");
    size_t n_values = 1;
    for (size_t i = 0; i < _dims.size(); i++)
        n_values *= _dims[i];

    return n_values;
}

// Retrieve a pointer to the tensor data.
void* TensorBase::data()
{
   return _data;
}

// Get a serialized buffer of the TensorBase data
std::string_view TensorBase::buf()
{
    /* This function returns a std::string_view of tensor
    data translated into a data buffer.  If the data buffer
    has not yet been created, the data buffer will be
    created before returning.
    */
    return std::string_view((char*)_data, _n_data_bytes());
}

// Validate inputs for a tensor
inline void TensorBase::_check_inputs(const void* src_data,
                                      const std::string& name,
                                      const std::vector<size_t>& dims)
{
    /* This function checks the validity of constructor
    inputs. This was taken out of the constructor to
    make the constructor actions more clear.
    */

    if (src_data == NULL) {
        throw SRRuntimeException("Must provide non-Null pointer to data.");
    }

    if (name.size() == 0) {
        throw SRRuntimeException("A name must be provided for the tensor");
    }

    if (name.compare(".meta") == 0) {
        throw SRRuntimeException(".meta is an internally reserved name "\
                                 "that is not allowed.");
    }

    if (dims.size() == 0) {
        throw SRRuntimeException("Must provide a dimensions vector with at "
                                 "least one dimension.");
    }

    std::vector<size_t>::const_iterator it = dims.cbegin();
    for ( ; it != dims.cend(); it++) {
        if (*it <= 0) {
            throw SRRuntimeException("All tensor dimensions "\
                                     "must be positive.");
        }
    }
}
