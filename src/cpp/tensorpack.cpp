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
#include "srexception.h"

using namespace SmartRedis;

// TensorPack copy constructor
TensorPack::TensorPack(const TensorPack& tp)
{
    if (this != &tp)
        _copy_tensor_inventory(tp);
}

// TensorPack copy assignment operator
TensorPack& TensorPack::operator=(const TensorPack& tp)
{
    // Check for self-assignment
    if (this == &tp)
        return *this;

    // Copy fields
    _all_tensors.clear();
    _tensorbase_inventory.clear();
    _copy_tensor_inventory(tp);
    return *this;
}

// Default TensorPack destructor
TensorPack::~TensorPack()
{
    typename TensorPack::tensorbase_iterator it = tensor_begin();
    for ( ; it != tensor_end(); it++)
        delete (*it);
}

// Add a tensor to the dataset
void TensorPack::add_tensor(const std::string& name,
                            void* data,
                            const std::vector<size_t>& dims,
                            const TensorType type,
                            const MemoryLayout mem_layout)
{
    // Check if it's already present
    if (tensor_exists(name)) {
        throw smart_runtime_error("The tensor " + std::string(name) +
                                  " already exists");
    }

    // Allocate memory for the tensor
    TensorBase* ptr = NULL;
    try {
        switch (type) {
            case TensorType::dbl:
                ptr = new Tensor<double>(name, data, dims, type, mem_layout);
                break;
            case TensorType::flt:
                ptr = new Tensor<float>(name, data, dims, type, mem_layout);
                break;
            case TensorType::int64:
                ptr = new Tensor<int64_t>(name, data, dims, type, mem_layout);
                break;
            case TensorType::int32:
                ptr = new Tensor<int32_t>(name, data, dims, type, mem_layout);
                break;
            case TensorType::int16:
                ptr = new Tensor<int16_t>(name, data, dims, type, mem_layout);
                break;
            case TensorType::int8:
                ptr = new Tensor<int8_t>(name, data, dims, type, mem_layout);
                break;
            case TensorType::uint16:
                ptr = new Tensor<uint16_t>(name, data, dims, type, mem_layout);
                break;
            case TensorType::uint8:
                ptr = new Tensor<uint8_t>(name, data, dims, type, mem_layout);
                break;
            default:
                throw smart_runtime_error("Unknown tensor type");
        }
    }
    catch (std::bad_alloc& e) {
        throw smart_bad_alloc("tensor data buffer");
    }

    // Add it
    add_tensor(ptr);
}

// Method to add a tensor object that has already been created on the heap.
// DO NOT add tensors allocated on the stack that may be deleted outside of
// the tensor pack. This function will cast the TensorBase to the correct
// Tensor<T> type.
void TensorPack::add_tensor(TensorBase* tensor)
{
    std::string name = tensor->name();

    if (name.size() == 0)
        throw smart_runtime_error("The tensor name must be nonempty.");

    _tensorbase_inventory[name] = tensor;
    _all_tensors.push_front(tensor);
}

// Return a TensorBase pointer based on name.
TensorBase* TensorPack::get_tensor(const std::string& name)
{
    return _tensorbase_inventory.at(name);
}

// Retrieve a pointer to the tensor data memory space
void* TensorPack::get_tensor_data(const std::string& name)
{
    TensorBase* ptr = _tensorbase_inventory.at(name);
    if (ptr == NULL)
        throw smart_runtime_error("Tensor not found: " + name);
    return ptr->data();
}

// Check whether a tensor with a given name exists in the TensorPack
bool TensorPack::tensor_exists(const std::string& name)
{
    return (_tensorbase_inventory.count(name) > 0);
}

// Retrieve an iterator pointing to the first Tensor
TensorPack::tensorbase_iterator TensorPack::tensor_begin()
{
    return _all_tensors.begin();
}

// Retrieve an iterator pointing to the last Tensor
TensorPack::tensorbase_iterator TensorPack::tensor_end()
{
    return _all_tensors.end();
}

// Retrieve a const iterator pointing to the first Tensor
TensorPack::const_tensorbase_iterator TensorPack::tensor_cbegin() const
{
    return _all_tensors.cbegin();
}

// Retrieve a const iterator pointing to the last Tensor
TensorPack::const_tensorbase_iterator TensorPack::tensor_cend() const
{
    return _all_tensors.cend();
}

// Copy the tensor inventory from one TensorPack to this TensorPack
void TensorPack::_copy_tensor_inventory(const TensorPack& tp)
{
    // Check for self-copy
    if (this == &tp)
        return;

    typename TensorPack::const_tensorbase_iterator it = tp.tensor_cbegin();
    for ( ; it != tp.tensor_cend(); it++) {
        TensorBase* ptr = (*it)->clone();
        if (ptr == NULL)
            throw smart_runtime_error("Invalid tensor found!");
        _all_tensors.push_front(ptr);
        _tensorbase_inventory[ptr->name()] = ptr;
    }
}
