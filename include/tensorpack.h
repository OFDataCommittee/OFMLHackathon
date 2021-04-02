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

#ifndef SMARTREDIS_TENSORPACK_H
#define SMARTREDIS_TENSORPACK_H

#include "stdlib.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <forward_list>
#include "tensor.h"
#include "tensorbase.h"
#include "tensorlist.h"

///@file

namespace SmartRedis {

class TensorPack;

/*!
*   \brief The TensorPack class is a container that
*          manages Tensors of multiple data types.
*/
class TensorPack
{
    public:

        /*!
        *   \brief Default TensorPack constructor
        */
        TensorPack() = default;

        /*!
        *   \brief TensorPack copy constructor
        *   \param tensorpack The TensorPack to be copied
        *                     for construction
        */
        TensorPack(const TensorPack& tensorpack);

        /*!
        *   \brief TensorPack default move constructor
        *   \param tensorpack The TensorPack to be moved
        *                     for construction
        */
        TensorPack(TensorPack&& tensorpack) = default;

        /*!
        *   \brief TensorPack copy assignment operator
        *   \param tensorpack The TensorPack to be copied
        *                     for assignment
        *   \returns TensorPack that has been assigned new values
        */
        TensorPack& operator=(const TensorPack& tensorpack);

        /*!
        *   \brief TensorPack default move assignment operator
        *   \param tensorpack TensorPack to be moved for assignment
        *   \returns TensorPack that has been assigned new values
        */
        TensorPack& operator=(TensorPack&& tensorpack) = default;

        /*!
        *   \brief Default TensorPack destructor
        */
        ~TensorPack() = default;

        /*!
        *   \brief Add a tensor to the dataset
        *   \param name The name used to reference the tensor
        *   \param data A c-ptr to the data of the tensor
        *   \param dims The dimensions of the data
        *   \param type The data type of the tensor
        *   \param mem_layout The memory layout of the data
        */
        void add_tensor(const std::string& name,
                        void* data,
                        const std::vector<size_t>& dims,
                        const TensorType type,
                        const MemoryLayout mem_layout);

        /*!
        *   \brief Method to add a tensor object that has
        *          already been created on the heap.
        *          DO NOT add tensors allocated on the
        *          stack that may be deleted outside of
        *          the tensor pack.  This function will cast
        *          the TensorBase to the correct Tensor<T> type.
        *   \param tensor Pointer to the tensor allocated on
        *                 the heap
        */
        void add_tensor(TensorBase* tensor);

        /*!
        *   \typedef An iterator type for iterating
        *            over all TensorBase items
        */
        typedef std::forward_list<TensorBase*>::iterator
                tensorbase_iterator;

        /*!
        *   \typedef A const iterator type for iterating
        *            over all TensorBase items
        */
        typedef std::forward_list<TensorBase*>::const_iterator
                const_tensorbase_iterator;

        /*!
        *   \brief Return a TensorBase pointer based on name.
        *   \param name The name used to reference the tensor
        *   \returns A pointer to the TensorBase object
        */
        TensorBase* get_tensor(const std::string& name);

        /*!
        *   \brief Return a pointer to the tensor data memory space
        *   \param name The name used to reference the tensor
        *   \returns A c-ptr to the tensor data
        */
        void* get_tensor_data(const std::string& name);

        /*!
        *   \brief Returns boolean indicating if tensor by
        *          that name exists in the TensorPack
        *   \param name The name used to reference the tensor
        *   \returns True if the name corresponds to a tensor
        *            in the TensorPack, otherwise False.
        */
        bool tensor_exists(const std::string& name);

        /*!
        *   \brief Returns an iterator pointing to the
        *          first TensorBase in the TensorPack
        *   \returns TensorPack iterator to the first
        *            TensorBase
        */
        tensorbase_iterator tensor_begin();

        /*!
        *   \brief Returns a const iterator pointing to the
        *          first TensorBase in the TensorPack
        *   \returns Const TensorPack iterator to the first
        *            TensorBase
        */
        const_tensorbase_iterator tensor_cbegin();


        /*!
        *   \brief Returns an iterator pointing to the
        *          past-the-end TensorBase in the TensorPack
        *   \returns TensorPack iterator to the past-the-end
        *            TensorBase
        */
        tensorbase_iterator tensor_end();

        /*!
        *   \brief Returns a const iterator pointing to the
        *          past-the-end TensorBase in the TensorPack
        *   \returns TensorPack const iterator to the
        *             past-the-end TensorBase
        */
        const_tensorbase_iterator tensor_cend();

    private:

        /*!
        *   \brief A forward list of all TensorBase
        *          to make iterating easier
        */
        std::forward_list<TensorBase*> _all_tensors;

        /*!
        *   \brief A map used to query and retrieve tensors.
        *          We can only return TensorBase
        *          to make iterating easier
        *          since the Tensors are templated.
        */
        std::unordered_map<std::string, TensorBase*> _tensorbase_inventory;


        /*!
        *   \brief TensorLists for memory management and
        *          operations on Tensor<double>
        *          to make iterating easier.
        */
        TensorList<double> _tensors_double;

        /*!
        *   \brief TensorLists for memory management and
        *          operations on Tensor<float>
        *          to make iterating easier.
        */
        TensorList<float> _tensors_float;

        /*!
        *   \brief TensorLists for memory management and
        *          operations on Tensor<int64_t>
        *          to make iterating easier.
        */
        TensorList<int64_t> _tensors_int64;

        /*!
        *   \brief TensorLists for memory management and
        *          operations on Tensor<int32_t>
        *          to make iterating easier.
        */
        TensorList<int32_t> _tensors_int32;

        /*!
        *   \brief TensorLists for memory management and
        *          operations on Tensor<int16_T>
        *          to make iterating easier.
        */
        TensorList<int16_t> _tensors_int16;

        /*!
        *   \brief TensorLists for memory management and
        *          operations on Tensor<int8_t>
        *          to make iterating easier.
        */
        TensorList<int8_t> _tensors_int8;

        /*!
        *   \brief TensorLists for memory management and
        *          operations on Tensor<uint16_t>
        *          to make iterating easier.
        */
        TensorList<uint16_t> _tensors_uint16;

        /*!
        *   \brief TensorLists for memory management and
        *          operations on Tensor<uint8_t>
        *          to make iterating easier.
        */
        TensorList<uint8_t> _tensors_uint8;

        /*!
        *   \brief Rebuild the tensor map and
        *          forward_list iterator
        */
        void _rebuild_tensor_inventory();

        /*!
        *   \brief Add a TensorList to the tensor
        *          inventory
        *   \param t_list The TensorList to add
        *   \tparam T The data type of the TensorList
        *           to add (e.g. double)
        */
        template <typename T>
        void _add_tensorlist_to_inventory(TensorList<T>& t_list);
};

} //namespace SmartRedis

#endif //SMARTREDIS_TENSORPACK_H
