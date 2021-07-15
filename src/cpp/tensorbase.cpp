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

#include "tensorbase.h"
#include <iostream>
using namespace SmartRedis;

TensorBase::TensorBase(const std::string& name,
                       void* data,
                       const std::vector<size_t>& dims,
                       const TensorType type,
                       const MemoryLayout mem_layout)
{
    /* The TensorBase constructor makes a copy of the
    name, type, and dims associated with the tensor.
    The provided data is copied into a memory space
    owned by the tensor.
    */

    this->_check_inputs(data, name, dims);
    this->_name = name;
    this->_type = type;
    this->_dims = dims;
}

TensorBase::TensorBase(const TensorBase& tb)
{
    /* This is the copy constructor for TensorBase.
    A deep copy of the tensor data is performed here.
    */
    this->_dims = std::vector<size_t>(tb._dims);
    this->_name = std::string(tb._name);
    this->_type = TensorType(tb._type);
}

TensorBase::TensorBase(TensorBase&& tb)
{
    /* This is the move constructor for TensorBase.
    */
    this->_name = std::move(tb._name);
    this->_type = std::move(tb._type);
    this->_dims = std::move(tb._dims);
    this->_data = tb._data;
    tb._data = 0;
}

TensorBase::~TensorBase()
{
    if(this->_data)
        free(this->_data);
}

TensorBase& TensorBase::operator=(const TensorBase& tb)
{
    /* This is the copy assignment operator for
    TensorBase.  A deep copy of the tensor
    data is performed.
    */
   if(this!=&tb) {
        this->_name = tb._name;
        this->_type = tb._type;
        this->_dims = tb._dims;
        if(this->_data)
            free(this->_data);
   }
   return *this;
}

TensorBase& TensorBase::operator=(TensorBase&& tb)
{
    /* This is the move assignment operator for
    TensorBase.
    */
    if(this!=&tb) {
        this->_name = std::move(tb._name);
        this->_type = std::move(tb._type);
        this->_dims = std::move(tb._dims);
        if(this->_data)
            free(this->_data);
        this->_data = tb._data;
        tb._data = 0;
    }
    return *this;
}

std::string TensorBase::name()
{
    /* Return the tensor name.
    */
    return this->_name;
}

TensorType TensorBase::type()
{
    /* Return the tensor type.
    */
   return this->_type;
}

std::string TensorBase::type_str(){
    /* Return the string version
    of the tensor type.
    */
    return TENSOR_STR_MAP.at(this->type());
}

std::vector<size_t> TensorBase::dims()
{
    /* Return the tensor dims
    */
   return this->_dims;
}

size_t TensorBase::num_values()
{
    /* Return the total number of values in the tensor
    */
    size_t n_values = this->_dims[0];
    for(size_t i=1; i<this->_dims.size(); i++) {
        n_values *= this->_dims[i];
    }
    return n_values;
}

void* TensorBase::data()
{
    /* This function returns a pointer to the
    tensor data.
    */
   return this->_data;
}

std::string_view TensorBase::buf()
{
    /* This function returns a std::string_view of tensor
    data translated into a data buffer.  If the data buffer
    has not yet been created, the data buffer will be
    created before returning.
    */
    return std::string_view((char*)this->_data,
                            this->_n_data_bytes());
}

inline void TensorBase::_check_inputs(const void* src_data,
                                      const std::string& name,
                                      const std::vector<size_t>& dims)
{
    /* This function checks the validity of constructor
    inputs. This was taken out of the constructor to
    make the constructor actions more clear.
    */

    if(!src_data)
        throw std::runtime_error("Must provide non-Null "\
                                 "pointer to data.");

    if(name.size()==0)
        throw std::runtime_error("A name must be "\
                                 "provided for the tensor");

    if(name.compare(".meta")==0)
        throw std::runtime_error(".META is an internally "\
                                 "reserved name that is not "\
                                 "allowed.");

    if(dims.size()==0)
        throw std::runtime_error("Must provide a dimensions "\
                                 "vector with at least one "\
                                 "dimension.");

    std::vector<size_t>::const_iterator it = dims.cbegin();
    std::vector<size_t>::const_iterator it_end = dims.cend();
    while(it!=it_end) {
        if((*it)<=0) {
            throw std::runtime_error("All tensor dimensions "\
                                     "must be positive.");
        }
        it++;
    }

    return;
}