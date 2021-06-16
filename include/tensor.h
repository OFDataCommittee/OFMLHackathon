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

#ifndef SMARTREDIS_TENSOR_H
#define SMARTREDIS_TENSOR_H

#include "stdlib.h"
#include <string>
#include <stdexcept>
#include "tensorbase.h"
#include "sharedmemorylist.h"

///@file

namespace SmartRedis {

/*!
*   \brief  The Tensor class stores, manages
*           and manipulates tensor data
*   \tparam T The data type of the Tensor
*/
template <class T>
class Tensor : public TensorBase
{
    public:

        /*!
        *   \brief Tensor constructor
        *   \param name The name used to reference the tensor
        *   \param data c-ptr to the source data for the tensor
        *   \param dims The dimensions of the tensor
        *   \param type The data type of the tensor
        *   \param mem_layout The memory layout of the source data
        */
        Tensor(const std::string& name,
               void* data,
               const std::vector<size_t>& dims,
               const TensorType type,
               const MemoryLayout mem_layout);

        /*!
        *   \brief Tensor copy constructor
        *   \param tensor The Tensor to copy for construction
        */
        Tensor(const Tensor<T>& tensor);

        /*!
        *   \brief Tensor move constructor
        *   \param tensor The Tensor to move for construction
        */
        Tensor(Tensor<T>&& tensor);

        /*!
        *   \brief Deep copy operator
        *   \details This method creates a new derived
        *            type Tensor and returns a TensorBase*
        *            pointer.  The new dervied type is
        *            allocated on the heap.  Contents
        *            are copied using the copy assignment
        *            operator for TensorBase and the derived
        *            type. This is meant
        *            to provide functionality to deep
        *            copy a Tensor when only a TensorBase
        *            object is possessed (i.e. a deep
        *            copy in a polymorphic capacity).
        *   \returns A pointer to dynamically allocated
        *            dervied type cast to parent TensorBase
        *            type.
        */
        virtual TensorBase* clone();

        /*!
        *   \brief Tensor destructor
        */
        virtual ~Tensor() = default;

        /*!
        *   \brief Tensor copy assignment operator
        *   \param tensor The Tensor to copy for assignment
        */
        Tensor<T>& operator=(const Tensor<T>& tensor);

        /*!
        *   \brief Tensor move assignment operator
        *   \param tensor The Tensor to move for assignment
        */
        Tensor<T>& operator=(Tensor<T>&& tensor);

        /*!
        *   \brief Get a pointer to a specificed memory
        *          view of the Tensor data
        *   \param mem_layout The MemoryLayout enum describing
        *          the layout of data view
        */
        virtual void* data_view(const MemoryLayout mem_layout);

        /*!
        *   \brief Fill a user provided memory space with
        *          values from tensor data
        *   \param data Pointer to the allocated memory space
        *   \param dims The dimensions of the memory space
        *   \param mem_layout The memory layout of the provided memory space
        */
        virtual void fill_mem_space(void* data,
                                    std::vector<size_t> dims,
                                    MemoryLayout mem_layout);

    protected:

    private:

        /*!
        *   \brief Function to copy values from nested memory
        *          structure to contiguous memory structure
        *   \details This function will copy the src_data,
        *            which is in a nested memory structure,
        *            to the dest_data memory space which is flat
        *            and contiguous.  The value returned by
        *            the first execution of this function will
        *            change the copy of dest_data and return
        *            a value that is not equal to the original
        *            source data value. As a result, the initial
        *            all of this function SHOULD NOT
        *            use the returned value.
        *   \param data The nested data to copy to the
        *               contiguous memory location
        *   \param dims The dimensions of src_data
        *   \param mem_layout The destination memory space (contiguous)
        *   \returns A pointer that is for recursive functionality
        *            only.  The initial caller SHOULD NOT use
        *            this pointer.
        */
        void* _copy_nested_to_contiguous(void* src_data,
                                         const size_t* dims,
                                         const size_t n_dims,
                                         void* dest_data);

        /*!
        *   \brief Function to copy from a flat, contiguous
        *          memory structure to a provided nested structure.
        *   \details The initial caller should provide an
        *            initial value of 0 for data_position.
        *   \param data The destination nested space for the flat data
        *   \param dims The dimensions of nested memory space
        *   \param n_dims The number of dimensions in dims
        *   \param data_position The current position of copying
        *                        the flat memory space
        *   \param tensor_data The flat memory structure data
        */
        void _fill_nested_mem_with_data(void* data,
                                        size_t* dims,
                                        size_t n_dims,
                                        size_t& data_position,
                                        void* tensor_data);

        /*!
        *   \brief Builds nested array structure to point
        *          to the provided flat, contiguous memory
        *          space.  The space is returned via the data
        *          input pointer.
        *   \details The initial caller SHOULD NOT use
        *            the return value.  The return value
        *            is for recursive value passing only.
        *   \param data A pointer to the pointer where
        *               data should be stored
        *   \param dims An array of integer dimension values
        *   \param n_dims The number of dimension left to
        *                 iterate through
        *   \param contiguous_mem The contiguous memory that
        *                         the nested structure points to
        *   \returns A pointer that is for recursive functionality
        *            only.  The initial caller SHOULD NOT use
        *            this pointer.
        */
        T* _build_nested_memory(void** data,
                                size_t* dims,
                                size_t n_dims,
                                T* contiguous_mem);

        /*!
        *   \brief Set the tensor data from a src memory location.
        *          This involves a memcpy to a contiguous array.
        *   \param src_data A pointer to the source data
        *                   being copied into the tensor
        *   \param dims The dimensions of the source data
        *   \param mem_layout The layout of the source data
        *                     memory structure
        */
        virtual void _set_tensor_data(void* src_data,
                                      const std::vector<size_t>& dims,
                                      const MemoryLayout mem_layout);

        /*!
        *   \brief This function will copy a fortran array
        *          memory space (column major) to a c-style
        *          memory space layout (row major)
        *   \param c_data A pointer to the row major memory space
        *   \param f_data A pointer to the col major memory space
        *   \param dims The dimensions of the tensor
        */
        void _f_to_c_memcpy(T* c_data,
                            T* f_data,
                            const std::vector<size_t>& dims);

        /*!
        *   \brief This function will copy a c-style array
        *          memory space (row major) to a fortran
        *          memory space layout (col major)
        *   \param f_data A pointer to the col major memory space
        *   \param c_data A pointer to the row major memory space
        *   \param dims The dimensions of the tensor
        */
        void _c_to_f_memcpy(T* f_data,
                            T* c_data,
                            const std::vector<size_t>& dims);

        /*!
        *   \brief This is a recursive function used to copy
        *          fortran column major memory to c-style row
        *          major memory
        *   \param c_data A pointer to the row major memory space
        *   \param f_data A pointer to the col major memory space
        *   \param dims The dimensions of the tensor
        *   \param dim_positions The current position in each
        *                        dimension
        *   \param current_dim The index of the current dimension
        */
        void _f_to_c(T* c_data,
                     T* f_data,
                     const std::vector<size_t>& dims,
                     std::vector<size_t> dim_positions,
                     size_t current_dim);

        /*!
        *   \brief This is a recursive function used to
        *          copy c-style row major memory to fortran
        *          column major memory
        *   \param f_data A pointer to the col major memory space
        *   \param c_data A pointer to the row major memory space
        *   \param dims The dimensions of the tensor
        *   \param dim_positions The current position in each
        *                        dimension
        *   \param current_dim The index of the current dimension
        */
        void _c_to_f(T* f_data,
                     T* c_data,
                     const std::vector<size_t>& dims,
                     std::vector<size_t> dim_positions,
                     size_t current_dim);

        /*!
        *   \brief Calculate the contiguous array position
        *          for a column major position
        *   \param dims The tensor dimensions
        *   \param dim_positions The current position for each
        *                        dimension
        *   \returns The contiguous memory index position
        */
        inline size_t _f_index(const std::vector<size_t>& dims,
                               const std::vector<size_t>& dim_positions);

        /*!
        *   \brief  Calculate the contiguous array position
        *           for a row major position
        *   \param dims The tensor dimensions
        *   \param dim_positions The current position for each dimension
        *   \returns The contiguous memory index position
        */
        inline size_t _c_index(const std::vector<size_t>& dims,
                               const std::vector<size_t>& dim_positions);

        /*!
        *   \brief Get the total number of bytes of the data
        *   \returns Total number of bytes of the data
        */
        virtual size_t _n_data_bytes();

        /*!
        *   \brief Memory allocated for c nested tensor memory views
        */
        SharedMemoryList<T*> _c_mem_views;

        /*!
        *   \brief Memory allocated for f nested tensor memory views
        */
        SharedMemoryList<T> _f_mem_views;
};

#include "tensor.tcc"

} //namespace SmartRedis

#endif //SMARTREDIS_TENSOR_H
