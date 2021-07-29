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

#include "tensorpack.h"

using namespace SmartRedis;

TensorPack::TensorPack(const TensorPack& tp)
{
    this->_copy_tensor_inventory(tp);
}

TensorPack& TensorPack::operator=(const TensorPack& tp)
{
    if(this!=&tp) {
        this->_all_tensors.clear();
        this->_tensorbase_inventory.clear();
        this->_copy_tensor_inventory(tp);
    }
    return *this;
}

TensorPack::~TensorPack()
{
    typename TensorPack::tensorbase_iterator it =
        this->tensor_begin();
    typename TensorPack::tensorbase_iterator it_end =
        this->tensor_end();

    while(it!=it_end) {
        delete (*it);
        it++;
    }
}

void TensorPack::add_tensor(const std::string& name,
                            void* data,
                            const std::vector<size_t>& dims,
                            const TensorType type,
                            const MemoryLayout mem_layout)
{

    if(this->tensor_exists(name))
        throw std::runtime_error("The tensor " +
                                 std::string(name) +
                                 " already exists");

    TensorBase* ptr;

    switch(type) {
        case TensorType::dbl :
            ptr = new Tensor<double>(name, data, dims,
                                    type, mem_layout);
            break;
        case TensorType::flt :
            ptr = new Tensor<float>(name, data, dims,
                                    type, mem_layout);
            break;
        case TensorType::int64 :
            ptr = new Tensor<int64_t>(name, data, dims,
                                      type, mem_layout);
            break;
        case TensorType::int32 :
            ptr = new Tensor<int32_t>(name, data, dims,
                                      type, mem_layout);
            break;
        case TensorType::int16 :
            ptr = new Tensor<int16_t>(name, data, dims,
                                     type, mem_layout);
            break;
        case TensorType::int8 :
            ptr = new Tensor<int8_t>(name, data, dims,
                                     type, mem_layout);
            break;
        case TensorType::uint16 :
            ptr = new Tensor<uint16_t>(name, data, dims,
                                       type, mem_layout);
            break;
        case TensorType::uint8 :
             ptr = new Tensor<uint8_t>(name, data, dims,
                                       type, mem_layout);
             break;
        default :
	  return; // FINDME: Handle this case better
    }
    this->add_tensor(ptr);
    return;
}

void TensorPack::add_tensor(TensorBase* tensor)
{
    std::string name = tensor->name();

    if(name.size()==0)
        throw std::runtime_error("The tensor name must "\
                                 "be greater than 0.");

    this->_tensorbase_inventory[name] = tensor;
    this->_all_tensors.push_front(tensor);
    return;
}

TensorBase* TensorPack::get_tensor(const std::string& name)
{
    TensorBase* ptr = this->_tensorbase_inventory.at(name);
    return ptr;
}

void* TensorPack::get_tensor_data(const std::string& name)
{
    TensorBase* ptr = this->_tensorbase_inventory.at(name);
    return ptr->data();
}

bool TensorPack::tensor_exists(const std::string& name)
{
    return (this->_tensorbase_inventory.count(name)>0);
}

TensorPack::tensorbase_iterator TensorPack::tensor_begin()
{
    return this->_all_tensors.begin();
}

TensorPack::tensorbase_iterator TensorPack::tensor_end()
{
    return this->_all_tensors.end();
}

TensorPack::const_tensorbase_iterator TensorPack::tensor_cbegin() const
{
    return this->_all_tensors.cbegin();
}

TensorPack::const_tensorbase_iterator TensorPack::tensor_cend() const
{
    return this->_all_tensors.cend();
}

void TensorPack::_copy_tensor_inventory(const TensorPack& tp)
{
    typename TensorPack::const_tensorbase_iterator it =
        tp.tensor_cbegin();
    typename TensorPack::const_tensorbase_iterator it_end =
        tp.tensor_cend();

    TensorBase* ptr;
    while(it!=it_end) {
        ptr = (*it)->clone();
        this->_all_tensors.push_front(ptr);
        this->_tensorbase_inventory[ptr->name()] = ptr;
        it++;
    }
    return;
}

