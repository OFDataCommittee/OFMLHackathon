/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

#ifndef SMARTREDIS_DATASET_H
#define SMARTREDIS_DATASET_H
#ifdef __cplusplus
#include "stdlib.h"
#include <string>
#include <vector>
#include "tensor.h"
#include "tensorpack.h"
#include "metadata.h"
#include "sharedmemorylist.h"
#include "sr_enums.h"

///@file

namespace SmartRedis{

class DataSet;

///@file
/*!
*   \brief The DataSet class aggregates tensors
*          and metadata into a nested data structure
*          for storage.
*   \details Tensors in the DataSet can be used in
*            Client commands such as Client.run_model()
*            and Client.run_script() as inputs or outputs
*            by prefixing the input or output tensor with
*            the DataSet name
*            (e.g. {dataset_name}.tensor_name).
*/
class DataSet
{
    public:

        /*!
        *   \brief DataSet constructor
        *   \param name The name used to reference the DataSet
        */
        DataSet(const std::string& name);

        /*!
        *   \brief DataSet copy constructor
        *   \param dataset The DataSet to copy
        */
        DataSet(const DataSet& dataset) = default;

        /*!
        *   \brief DataSet copy assignment operator
        *   \param dataset The DataSet to copy and assign
        */
        DataSet& operator=(const DataSet& dataset) = default;

        /*!
        *   \brief DataSet move constructor
        *   \param dataset The DataSet to move
        */
        DataSet(DataSet&& dataset) = default;

        /*!
        *   \brief DataSet move assignment operator
        *   \param dataset The DataSet to move and assign
        */
        DataSet& operator=(DataSet&& dataset) = default;

        /*!
        *   \brief Add a tensor to the DataSet.
        *   \param name The name used to reference the tensor
        *               within the DataSet
        *   \param data The tensor data
        *   \param dims The number of elements in each dimension of the tensor
        *   \param type The data type of the provided tensor data
        *   \param mem_layout The memory layout of the provided tensor data
        *   \throw SmartRedis::Exception if add_tensor operation fails
        */
        void add_tensor(const std::string& name,
                        void* data,
                        const std::vector<size_t>& dims,
                        const SRTensorType type,
                        const SRMemoryLayout mem_layout);

        /*!
        *   \brief Append a metadata scalar value to a field in the DataSet.
        *          If the field does not exist, it will be created.
        *          For string scalars, use add_meta_string.
        *   \param name The name for the metadata field
        *   \param data The scalar data to be appended to the metadata field
        *   \param type The data type of the scalar data to be appended to
        *               the metadata field
        *   \throw SmartRedis::Exception if add_meta_scalar operation fails
        */
        void add_meta_scalar(const std::string& name,
                             const void* data,
                             const SRMetaDataType type);

        /*!
        *   \brief Append a metadata string value to a field in the DataSet.
        *          If the field does not exist, it will be created.
        *   \param name The name for the metadata field
        *   \param data The string to be appended to the metadata field
        *   \throw SmartRedis::Exception if add_meta_string operation fails
        */
        void add_meta_string(const std::string& name,
                             const std::string& data);

        /*!
        *   \brief Get the tensor data, dimensions, and type for the tensor
        *          in the DataSet. This function will allocate and retain
        *          management of the memory for the tensor data.
        *   \details The memory of the data pointer is valid until the
        *            DataSet is destroyed. This method is meant to be used
        *            when the dimensions and type of the tensor are unknown
        *            or the user does not want to manage memory.  However,
        *            given that the memory associated with the return data is
        *            valid until DataSet destruction, this method should not
        *            be used repeatedly for large tensor data. Instead, it
        *            is recommended that the user use unpack_tensor() for
        *            large tensor data and to limit memory use by the DataSet.
        *   \param name The name used to reference the tensor in the DataSet
        *   \param data Receives data for the tensor, allocated by the library
        *   \param dims Receives the number of elements in each dimension of
        *               the retrieved tensor data
        *   \param type Receives the data type for the tensor
        *   \param mem_layout The memory layout to which retrieved tensor data
        *                     should conform
        *   \throw SmartRedis::Exception if tensor retrieval fails
        */
        void get_tensor(const std::string& name,
                        void*& data,
                        std::vector<size_t>& dims,
                        SRTensorType& type,
                        const SRMemoryLayout mem_layout);

        /*!
        *   \brief Get the tensor data, dimensions, and type for the tensor
        *          in the DataSet. This function will allocate and retain
        *          management of the memory for the tensor data.
        *          This is a c-style interface for the tensor dimensions
        *   \details The memory of the data pointer is valid until the
        *            DataSet is destroyed. This method is meant to be used
        *            when the dimensions and type of the tensor are unknown
        *            or the user does not want to manage memory.  However,
        *            given that the memory associated with the return data is
        *            valid until DataSet destruction, this method should not
        *            be used repeatedly for large tensor data. Instead, it
        *            is recommended that the user use unpack_tensor() for
        *            large tensor data and to limit memory use by the DataSet.
        *   \param name The name used to reference the tensor in the DataSet
        *   \param data Receives data for the tensor, allocated by the library
        *   \param dims Receives the number of elements in each dimension of
        *               the retrieved tensor data
        *   \param n_dims Receives the number of dimensions of tensor data
        *                 retrieved
        *   \param type Receives the data type for the tensor
        *   \param mem_layout The memory layout to which retrieved tensor data
        *                     should conform
        *   \throw SmartRedis::Exception if tensor retrieval fails
        */
        void get_tensor(const std::string& name,
                        void*& data,
                        size_t*& dims,
                        size_t& n_dims,
                        SRTensorType& type,
                        const SRMemoryLayout mem_layout);

        /*!
        *   \brief Retrieve tensor data to a caller-supplied buffer.
        *          This method is the most memory efficient way
        *          to retrieve tensor data from a DataSet.
        *   \param name The name used to reference the tensor
        *               in the DataSet
        *   \param data Receives data for the tensor, allocated by the library
        *   \param dims The number of elemnents in each dimension of the
        *               provided data buffer
        *   \param type The tensor datatype for the provided data buffer
        *   \param mem_layout The memory layout for the provided data buffer
        *   \throw SmartRedis::Exception if tensor retrieval fails
        */
        void unpack_tensor(const std::string& name,
                           void* data,
                           const std::vector<size_t>& dims,
                           const SRTensorType type,
                           const SRMemoryLayout mem_layout);

        /*!
        *   \brief Retrieve metadata scalar field values from the DataSet.
        *   \details The memory of the data pointer is valid until the
        *            DataSet is destroyed.
        *   \param name The name for the metadata field in the DataSet
        *   \param data Receives scalar data from the metadata field
        *   \param length Receives the number of returned scalar data values
        *   \param type Receives the number data type of the returned
        *               scalar data
        *   \throw SmartRedis::Exception if metadata retrieval fails
        */
        void get_meta_scalars(const std::string& name,
                              void*& data,
                              size_t& length,
                              SRMetaDataType& type);

        /*!
        *   \brief Retrieve metadata string field values from the DataSet.
        *          Because standard C++ containers are used,
        *          memory management is handled by the returned
        *          std::vector<std::string>.
        *   \param name The name of the metadata string field
        *   \return The strings associated with the metadata field, or
        *           an empty vector if no field matches the supplied
        *           metadata field name
        *   \throw SmartRedis::Exception if metadata retrieval fails
        */
        std::vector<std::string> get_meta_strings(const std::string& name);

        /*!
        *   \brief Retrieve metadata string field values from the DataSet.
        *   \details The memory of the data pointer is valid until the
        *            DataSet is destroyed.
        *   \param name The name of the metadata field in the DataSet
        *   \param data Receives an array of strings associated with the
        *               metadata field
        *   \param n_strings Receives the number of strings found that
        *                    match the supplied metadata field name
        *   \param lengths Receives an array of the lengths of the strings
        *                  found that match the supplied metadata field name
        *   \throw SmartRedis::Exception if metadata retrieval fails
        */
        void get_meta_strings(const std::string& name,
                              char**& data,
                              size_t& n_strings,
                              size_t*& lengths);

        /*!
        *   \brief Check whether the dataset contains a field
        *   \param field_name The name of the field to check
        *   \returns True iff the DataSet contains the field
        */
        bool has_field(const std::string& field_name);


        /*!
        *   \brief Clear all entries from a DataSet field.
        *   \param field_name The name of the field to clear
        */
        void clear_field(const std::string& field_name);

        /*!
        *   \brief Retrieve the names of tensors in the DataSet
        *   \returns The name of the tensors in the DataSet
        *   \throw SmartRedis::Exception if metadata retrieval fails
        */
        std::vector<std::string> get_tensor_names();

        /*!
        *   \brief Retrieve the name of the DataSet
        *   \returns The name of the DataSet
        */
        std::string get_name() const { return _dsname; }

        /*!
        *   \brief Change the name for the DataSet
        *   \param name The name for the DataSet
        */
        void set_name(std::string name) { _dsname = name; }

        friend class Client;
        friend class PyDataset;

    protected:

        /*!
        *   \brief Iterator for Tensor in the dataset
        */
        typedef TensorPack::tensorbase_iterator tensor_iterator;

        /*!
        *   \brief Const iterator for Tensor in the dataset
        */
        typedef TensorPack::const_tensorbase_iterator const_tensor_iterator;

        /*!
        *   \brief Get an iterator pointing to the first tensor in the DataSet
        *   \returns tensor_iterator to the first Tensor
        */
        tensor_iterator tensor_begin();

        /*!
        *   \brief Get a const iterator pointing to the first
        *          tensor in the DataSet
        *   \returns const_tensor_iterator to the first Tensor
        */
        const_tensor_iterator tensor_cbegin();

        /*!
        *   \brief Get an iterator pointing to the past-the-end tensor
        *   \returns tensor_iterator to the past-the-end Tensor
        */
        tensor_iterator tensor_end();

        /*!
        *   \brief Returns an const iterator pointing to the
        *          past-the-end tensor
        *   \returns const_tensor_iterator to the past-the-end Tensor
        */
        const_tensor_iterator tensor_cend();

        /*!
        *   \brief Retrieve the data type of a Tensor in the DataSet
        *   \param name The name of the tensor
        *   \returns The data type for the tensor
        */
        SRTensorType get_tensor_type(const std::string& name);

        /*!
        *   \brief Returns a vector of std::pair with
        *          the field name and the field serialization
        *          for all fields in the MetaData set.
        *   \returns std::pair<std::string, std::string> containing
        *            the field name and the field serialization.
        */
        std::vector<std::pair<std::string, std::string>>
            get_metadata_serialization_map();

        /*!
        *   \brief Add a Tensor (not yet allocated) to the TensorPack
        *   \param name The name of the Tensor
        *   \param data The tensor data
        *   \param dims The number of elements in each dimension of the tensor
        *   \param type The data type for the tensor
        *   \param mem_layout The memory layout for the provided tensor data
        */
        void _add_to_tensorpack(const std::string& name,
                                void* data,
                                const std::vector<size_t>& dims,
                                const SRTensorType type,
                                const SRMemoryLayout mem_layout);

        /*!
        *   \brief Add a serialized field to the DataSet
        *   \param name The name of the field
        *   \param buf The buffer used for object construction
        *   \param buf_size The length of the buffer
        */
        void _add_serialized_field(const std::string& name,
                                   char* buf,
                                   size_t buf_size);

        /*!
        *   \brief Retrieve the tensor from the DataSet and return
        *          a TensorBase object that can be used to return
        *          tensor information to the user.  The returned
        *          TensorBase object has been dynamically allocated,
        *          but not yet tracked for memory management in
        *          any object.
        *   \details The TensorBase object returned will always
        *            have a MemoryLayout::contiguous layout.
        *   \param name  The name used to reference the tensor
        *   \returns A TensorBase object.
        */
        TensorBase* _get_tensorbase_obj(const std::string& name);

    private:

        /*!
        *  \brief The name of the DataSet
        */
        std::string _dsname;

        /*!
        *   \brief A repository for all metadata associated with this DataSet
        */
        MetaData _metadata;

        /*!
        *   \brief A repository for all tensor associated with this DataSet
        */
        TensorPack _tensorpack;

        /*!
        *   \brief Throw an exception if a tensor does not exist
        *   \throw RuntimeException if the tensor is not in the DataSet
        */
        inline void _enforce_tensor_exists(const std::string& name);

        /*!
        *   \brief SharedMemoryList to manage memory associated
        *          with tensor dimensions from tensor retrieval
        */
        SharedMemoryList<size_t> _dim_queries;

        /*!
        *  \brief The _tensor_pack memory is not for querying
        *         by name, but is used to manage memory associated
        *         with get_tensor() function calls.
        */
        TensorPack _tensor_memory;

};

} //namespace SmartRedis

#endif
#endif //SMARTREDIS_DATASET_H
