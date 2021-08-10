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
#include "enums/cpp_tensor_type.h"
#include "enums/cpp_memory_layout.h"
#include "enums/cpp_metadata_type.h"

///@file

namespace SmartRedis{

class DataSet;

///@file
/*!
*   \brief The DataSet class aggregates tensors
*          and metdata into a nested data structure
*          for storage.
*   \details Tensors in the DataSet can be used in
*            Client commands like Client.run_model()
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
        *   \param data A c-ptr to the tensor data memory location
        *   \param dims The dimensions of the tensor
        *   \param type The data type of the tensor data
        *   \param mem_layout The MemoryLayout enum describing the
        *                     layout of the provided tensor
        *                     data
        */
        void add_tensor(const std::string& name,
                        void* data,
                        const std::vector<size_t>& dims,
                        const TensorType type,
                        const MemoryLayout mem_layout);

        /*!
        *   \brief Add metadata scalar field (non-string)
        *          with value to the DataSet.  If the
        *          field does not exist, it will be created.
        *          If the field exists, the value
        *          will be appended to existing field.
        *   \param name The name used to reference the metadata
        *               field
        *   \param data A c-ptr to the metadata field data
        *   \param type The data type of the metadata
        */
        void add_meta_scalar(const std::string& name,
                             const void* data,
                             const MetaDataType type);

        /*!
        *   \brief Add metadata string field with value
        *          to the DataSet.  If the field
        *          does not exist, it will be created.
        *          If the field exists the value will
        *          be appended to existing field.
        *   \param name The name used to reference the metadata
        *               field
        *   \param data The string to add to the field
        */
        void add_meta_string(const std::string& name,
                             const std::string& data);

        /*!
        *   \brief Get the tensor data, dimensions,
        *          and type for the tensor in the DataSet.
        *          This function will allocate and retain
        *          management of the memory for the tensor
        *          data.
        *   \details The memory of the data pointer is valid
        *            until the DataSet is destroyed. This method
        *            is meant to be used when the dimensions and
        *            type of the tensor are unknown or the user does
        *            not want to manage memory.  However, given that
        *            the memory associated with the return data is
        *            valid until DataSet destruction, this method
        *            should not be used repeatedly for large tensor
        *            data.  Instead  it is recommended that the user
        *            use unpack_tensor() for large tensor data and
        *            to limit memory use by the DataSet.
        *   \param name The name used to reference the tensor in
        *               the DataSet
        *   \param data A c-ptr reference that will be pointed to
        *               newly allocated memory
        *   \param dims A reference to a dimensions vector
        *               that will be filled with the retrieved
        *               tensor dimensions
        *   \param type A reference to a TensorType enum that will
        *               be set to the value of the retrieved
        *               tensor type
        *   \param mem_layout The MemoryLayout that the newly
        *                     allocated memory should conform to
        */
        void get_tensor(const std::string& name,
                        void*& data,
                        std::vector<size_t>& dims,
                        TensorType& type,
                        const MemoryLayout mem_layout);

        /*!
        *   \brief Get the tensor data, dimensions,
        *          and type for the tensor in the DataSet.
        *          This function will allocate and retain
        *          management of the memory for the tensor
        *          data.  This is a c-style
        *          interface for the tensor dimensions.  Another
        *          function exists for std::vector dimensions.
        *   \details The memory of the data pointer is valid
        *            until the DataSet is destroyed. This method
        *            is meant to be used when the dimensions and
        *            type of the tensor are unknown or the user does
        *            not want to manage memory.  However, given that
        *            the memory associated with the return data is
        *            valid until DataSet destruction, this method
        *            should not be used repeatedly for large tensor
        *            data.  Instead  it is recommended that the user
        *            use unpack_tensor() for large tensor data and
        *            to limit memory use by the DataSet.
        *   \param name The name used to reference the tensor in
        *               the DataSet
        *   \param data A c-ptr reference that will be pointed to
        *               newly allocated memory
        *   \param dims A reference to a dimensions vector
        *               that will be filled with the retrieved
        *               tensor dimensions
        *   \param type A reference to a TensorType enum that will
        *               be set to the value of the retrieved
        *               tensor type
        *   \param mem_layout The MemoryLayout that the newly
        *                     allocated memory should conform to
        */
        void get_tensor(const std::string& name,
                        void*& data,
                        size_t*& dims,
                        size_t& n_dims,
                        TensorType& type,
                        const MemoryLayout mem_layout);

        /*!
        *   \brief Get tensor data and fill an already allocated
        *          array memory space that has the specified
        *          MemoryLayout.  The provided type and dimensions
        *          are checked against retrieved values to ensure
        *          the provided memory space is sufficient.  This
        *          method is the most memory efficient way
        *          to retrieve tensor data from a DataSet.
        *   \param name The name used to reference the tensor
        *               in the DataSet
        *   \param data A c-ptr to the memory space to be filled
        *               with tensor data
        *   \param dims The dimensions of the memory space
        *   \param type The TensorType matching the data
        *               type of the memory space
        *   \param mem_layout The MemoryLayout of the provided
        *               memory space.
        */
        void unpack_tensor(const std::string& name,
                           void* data,
                           const std::vector<size_t>& dims,
                           const TensorType type,
                           const MemoryLayout mem_layout);

        /*!
        *   \brief Get the metadata scalar field values
        *          from the DataSet.  The data pointer
        *          reference will be pointed to newly
        *          allocated memory that will contain
        *          all values in the metadata field.
        *          The length variable will be set to
        *          the number of entries in the allocated
        *          memory space to allow for iteration over
        *          the values.  The TensorType enum
        *          will be set to the type of the MetaData
        *          field.
        *   \param name The name used to reference the metadata
        *               field in the DataSet
        *   \param data A c-ptr to the memory space to be filled
        *               with the metadata field
        *   \param length The number of values in the metadata
        *                 field
        *   \param type The MetadataType enum describing
        *               the data type of the metadata field
        */
        void get_meta_scalars(const std::string& name,
                              void*& data,
                              size_t& length,
                              MetaDataType& type);

        /*!
        *   \brief Get the strings in a metadata string field.
        *          Because standard C++ containers are used,
        *          memory management is handled by the returned
        *          std::vector<std::string>.
        *   \param name The name of the metadata string field
        */
        std::vector<std::string> get_meta_strings(const std::string& name);

        /*!
        *   \brief Get the metadata string field values
        *          from the DataSet.  The data pointer
        *          reference will be pointed to newly
        *          allocated memory that will contain
        *          all values in the metadata field.
        *          The n_strings input variable
        *          reference will be set to the number of
        *          strings in the field.  The lengths
        *          c-ptr variable will be pointed to a
        *          new memory space that contains
        *          the length of each field string.  The
        *          memory for the data and lengths pointers
        *          will be managed by the DataSet object
        *          and remain valid until the DataSet
        *          object is destroyed.
        *   \param name The name used to reference the metadata
        *               field in the DataSet
        *   \param data A c-ptr that will be pointed to a memory space
        *               filled with the metadata field strings
        *   \param n_strings A reference to a size_t variable
        *                    that will be set to the number of
        *                    strings in the field
        *   \param lengths A c-ptr that will be pointed to a
        *                  memory space that contains the length
        *                  of each field string
        */
        void get_meta_strings(const std::string& name,
                              char**& data,
                              size_t& n_strings,
                              size_t*& lengths);

        /*!
        *   \brief This function checks if the DataSet has a
        *          field
        *   \param field_name The name of the field to check
        *   \returns Boolean indicating if the DataSet has
        *            the field.
        */
        bool has_field(const std::string& field_name);


        /*!
        *   \brief This function clears all entries in a
        *          DataSet field.
        *   \param field_name The name of the field to clear
        */
        void clear_field(const std::string& field_name);

        /*!
        *   \brief Retrieve the names of the tensors in the
        *          DataSet
        *   \returns The name of the tensors in the DataSet
        */
        std::vector<std::string> get_tensor_names();

        /*!
        *  \brief The name of the DataSet
        */
        std::string name;

        friend class Client;
        friend class PyDataset;

    protected:

        /*!
        *   \typedef Iterator for Tensor in the dataset
        */
        typedef TensorPack::tensorbase_iterator tensor_iterator;

        /*!
        *   \typedef Const iterator for Tensor in the dataset
        */
        typedef TensorPack::const_tensorbase_iterator const_tensor_iterator;

        /*!
        *   \brief Returns an iterator pointing to the
        *          first Tensor in the DataSet
        *   \returns tensor_iterator to the first Tensor
        */
        tensor_iterator tensor_begin();

        /*!
        *   \brief Returns a const iterator pointing to the
        *          first Tensor in the DataSet
        *   \returns const_tensor_iterator to the first Tensor
        */
        const_tensor_iterator tensor_cbegin();

        /*!
        *   \brief Returns an iterator pointing to
        *          the past-the-end Tensor
        *   \returns tensor_iterator to the past-the-end Tensor
        */
        tensor_iterator tensor_end();

        /*!
        *   \brief Returns an const iterator pointing to
        *          the past-the-end Tensor
        *   \returns const_tensor_iterator to the past-the-end Tensor
        */
        const_tensor_iterator tensor_cend();

        /*!
        *   \brief Get the Tensor type of the Tensor
        *   \param name The name of the Tensor
        *   \returns The Tensor's TensorType
        */
        TensorType get_tensor_type(const std::string& name);

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
        *   \param data A c-ptr to the beginning of the tensor data
        *   \param dims The dimensions of the tensor
        *   \param type The data type of the tensor
        *   \param mem_layout The memory layout of the provided
        *                     tensor data
        */
        void _add_to_tensorpack(const std::string& name,
                                void* data,
                                const std::vector<size_t>& dims,
                                const TensorType type,
                                const MemoryLayout mem_layout);

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
        *   \brief MetaData object containing all metadata
        */
        MetaData _metadata;

        /*!
        *   \brief TensorPack object containing all Tensors
        *          in DataSet
        */
        TensorPack _tensorpack;

        /*!
        *   \brief Check and enforce that a tensor must exist or
        *          throw an error.
        *   \throw std::runtime_error if the tensor is not
        *          in the DataSet.
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
