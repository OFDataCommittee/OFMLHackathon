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

#ifndef SMARTREDIS_TENSORBASE_H
#define SMARTREDIS_TENSORBASE_H

#include "stdlib.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <string_view>
#include <stdexcept>
#include "enums/cpp_tensor_type.h"
#include "enums/cpp_memory_layout.h"

///@file

namespace SmartRedis {

/* The strings and unordered map below is used for
fast conversion between RedisAI string and enum
value
*/
// Numeric data type of tensor elements that are allowed
static std::string DATATYPE_TENSOR_STR_FLOAT = "FLOAT";
static std::string DATATYPE_TENSOR_STR_DOUBLE = "DOUBLE";
static std::string DATATYPE_TENSOR_STR_INT8 = "INT8";
static std::string DATATYPE_TENSOR_STR_INT16 = "INT16";
static std::string DATATYPE_TENSOR_STR_INT32 = "INT32";
static std::string DATATYPE_TENSOR_STR_INT64 = "INT64";
static std::string DATATYPE_TENSOR_STR_UINT8 = "UINT8";
static std::string DATATYPE_TENSOR_STR_UINT16 = "UINT16";

static const std::unordered_map<std::string, TensorType>
    TENSOR_TYPE_MAP{
        {DATATYPE_TENSOR_STR_DOUBLE, TensorType::dbl},
        {DATATYPE_TENSOR_STR_FLOAT, TensorType::flt},
        {DATATYPE_TENSOR_STR_INT64, TensorType::int64},
        {DATATYPE_TENSOR_STR_INT32, TensorType::int32},
        {DATATYPE_TENSOR_STR_INT16, TensorType::int16},
        {DATATYPE_TENSOR_STR_INT8, TensorType::int8},
        {DATATYPE_TENSOR_STR_UINT16, TensorType::uint16},
        {DATATYPE_TENSOR_STR_UINT8, TensorType::uint8} };

static const std::unordered_map<TensorType, std::string>
    TENSOR_STR_MAP{
        {TensorType::dbl, DATATYPE_TENSOR_STR_DOUBLE},
        {TensorType::flt, DATATYPE_TENSOR_STR_FLOAT},
        {TensorType::int64, DATATYPE_TENSOR_STR_INT64},
        {TensorType::int32, DATATYPE_TENSOR_STR_INT32},
        {TensorType::int16, DATATYPE_TENSOR_STR_INT16},
        {TensorType::int8, DATATYPE_TENSOR_STR_INT8},
        {TensorType::uint16, DATATYPE_TENSOR_STR_UINT16},
        {TensorType::uint8, DATATYPE_TENSOR_STR_UINT8} };

class TensorBase;

/*!
*   \brief  The TensorBase class is a base class that
*           defines an interface for templated classes
*           that inherit from TensorBase.
*/
class TensorBase{

    public:

        /*!
        *   \brief TensorBase constructor
        *   \param name The name used to reference the tensor
        *   \param data c-ptr to the source data for the tensor
        *   \param dims The dimensions of the tensor
        *   \param type The data type of the tensor
        *   \param mem_layout The memory layout of the source data
        */
        TensorBase(const std::string& name,
                   void* data,
                   const std::vector<size_t>& dims,
                   const TensorType type,
                   const MemoryLayout mem_layout);

        /*!
        *   \brief TensorBase copy constructor
        *   \param tb The TensorBase to copy for construction
        */
        TensorBase(const TensorBase& tb);

        /*!
        *   \brief TensorBase move constructor
        *   \param tb The TensorBase to move for construction
        */
        TensorBase(TensorBase&& tb);

        /*!
        *   \brief Deep copy operator
        *   \details This method creates a new derived
        *            type Tensor and returns a TensorBase*
        *            pointer.  The new dervied type is
        *            allocated on the heap.  Contents
        *            are copied using the copy assignment
        *            operator for the derived type. This is meant
        *            to provide functionality to deep
        *            copy a Tensor when only a TensorBase
        *            object is possessed (i.e. a deep
        *            copy in a polymorphic capacity).
        *   \returns A pointer to dynamically allocated
        *            dervied type cast to parent TensorBase
        *            type.
        */
        virtual TensorBase* clone() = 0;

        /*!
        *   \brief TensorBase destructor
        */
        virtual ~TensorBase();

        /*!
        *   \brief TensorBase copy assignment operator
        *   \param tb The TensorBase to copy for assignment
        */
        TensorBase& operator=(const TensorBase& tb);

        /*!
        *   \brief TensorBase move assignment operator
        *   \param tb The TensorBase to move for assignment
        */
        TensorBase& operator=(TensorBase&& tb);

        /*!
        *   \brief Retrieve the name of the TensorBase
        *   \returns The name of the TensorBase
        */
        std::string name();

        /*!
        *   \brief Retrieve the type of the TensorBase
        *   \returns The type of the TensorBase
        */
        TensorType type();

        /*!
        *   \brief Retrieve a string representation of
        *          the TensorBase
        *   \returns A string representation of the TensorBase
        */
        std::string type_str();

        /*!
        *   \brief Retrieve the dimensions of the TensorBase
        *   \returns TensorBase dimensions
        */
        std::vector<size_t> dims();

        /*!
        *   \brief Retrieve number of values in the TensorBase
        *   \returns The number values in the TensorBase
        */
        size_t num_values();

        /*!
        *   \brief Retrieve a pointer to the TensorBase data
        *          memory
        *   \returns A pointer to the TenorBase data memory
        */
        void* data();

        /*!
        *   \brief Get a serialized buffer of the TensorBase
        *          data
        *   \returns A std::string_view buffer of TensorBase
        *          data
        */
        virtual std::string_view buf();

        /*!
        *   \brief Get a pointer to a specificed memory
        *          view of the TensorBase data
        *   \param mem_layout The MemoryLayout enum describing
        *          the layout of data view
        */
        virtual void* data_view(const MemoryLayout mem_layout) = 0;

        /*!
        *   \brief Fill a user provided memory space with
        *          values from tensor data
        *   \param data Pointer to the allocated memory space
        *   \param dims The dimensions of the memory space
        *   \param mem_layout The memory layout of the provided memory space
        */
        virtual void fill_mem_space(void* data,
                                    std::vector<size_t> dims,
                                    MemoryLayout mem_layout) = 0;


        protected:

        /*!
        *   \brief TensorBase name
        */
        std::string _name;

        /*!
        *   \brief TensorBase type
        */
        TensorType _type;

        /*!
        *   \brief TensorBase dims
        */
        std::vector<size_t> _dims;

        /*!
        *   \brief Pointer to the data memory space
        */
        void* _data;

        //TODO implement this
        //! Function to copy tensor data into this tensor data
        //virtual void _copy_data(void* data /*!< A c-ptr to the data to copy*/,
        //                        std::vector<int> dims /*! The dimensions of the data to copy*/
        //                        ) = 0;

        private:

        /*!
        *   \brief Pointer to the data memory space
        *   \param src_data A pointer to the data source for the tensor
        *   \param name The name used to reference the tensor
        *   \param dims The dimensions of the data
        */
        inline void _check_inputs(const void* src_data,
                                  const std::string& name,
                                  const std::vector<size_t>& dims);

        /*!
        *   \brief Set the tensor data from a src memory location
        *   \param src_data A pointer to the data source for the tensor
        *   \param dims The dimensions of the data
        *   \param mem_layout The memory layout of the source data
        */
        virtual void _set_tensor_data(void* src_data,
                                      const std::vector<size_t>& dims,
                                      const MemoryLayout mem_layout) = 0;

        /*!
        *   \brief Get the total number of bytes of the data
        *   \returns Total number of bytes of the data
        */
        virtual size_t _n_data_bytes() = 0;
};

} //namespace SmartRedis

#endif //SMARTREDIS_TENSORBASE_H
