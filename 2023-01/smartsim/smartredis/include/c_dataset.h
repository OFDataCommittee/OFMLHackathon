
/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

#ifndef SMARTREDIS_C_DATASET_H
#define SMARTREDIS_C_DATASET_H

#include "dataset.h"
#include "sr_enums.h"
#include "srexception.h"

///@file
///\brief C-wrappers for the C++ DataSet class

#ifdef __cplusplus
extern "C" {
#endif

/*!
*   \brief C-DataSet constructor
*   \param name The name of the dataset
*   \param name_length The length of the dataset name string,
*                      excluding null terminating character
*   \param new_dataset Receives the new dataset
*   \return Returns SRNoError on success or an error code on failure
*/
SRError CDataSet(const char* name,
                 const size_t name_length,
                 void** new_dataset);

/*!
*   \brief C-DataSet destructor
*   \param dataset A pointer to the dataset to release. The dataset is
*                  set to NULL on completion
*   \return Returns SRNoError on success or an error code on failure
*/
SRError DeallocateeDataSet(void** dataset);

/*!
*   \brief Add a tensor to the DataSet.
*   \param dataset The dataset to use for this operation
*   \param name The name by which this tensor should be referenced
*               in the DataSet
*   \param name_length The length of the dataset name string,
*                      excluding null terminating character
*   \param data Tensor data to be stored in the dataset
*   \param dims The number of elements in each dimension of the tensor
*   \param n_dims The number of dimensions for the tensor
*   \param type The data type of the tensor data
*   \param mem_layout Memory layout of the provided tensor data
*   \return Returns SRNoError on success or an error code on failure
*/
SRError add_tensor(void* dataset,
                   const char* name,
                   const size_t name_length,
                   void* data,
                   const size_t* dims,
                   const size_t n_dims,
                   const SRTensorType type,
                   const SRMemoryLayout mem_layout);

/*!
*   \brief Append a metadata scalar value to a field in the DataSet.
*          If the field does not exist, it will be created.
*          For string scalars, use add_meta_string.
*   \param dataset The dataset to use for this operation
*   \param name The name for the metadata field
*   \param name_length The length of the dataset name string,
*                      excluding null terminating character
*   \param data The scalar data to be appended to the metadata field
*   \param type The data type of the metadata scalar
*   \return Returns SRNoError on success or an error code on failure
*/
SRError add_meta_scalar(void* dataset,
                        const char* name,
                        const size_t name_length,
                        const void* data,
                        const SRMetaDataType type);


/*!
*   \brief Append a metadata string to a field the DataSet.  If the field
*          does not exist, it will be created.
*   \param dataset The dataset to use for this operation
*   \param name The name for the metadata field
*   \param name_length The length of the dataset name string,
*                      excluding null terminating character
*   \param data The string to add to the field
*   \param data_length The length of the metadata string value,
*                      excluding null terminating character
*   \return Returns SRNoError on success or an error code on failure
*/
SRError add_meta_string(void* dataset,
                        const char* name,
                        const size_t name_length,
                        const char* data,
                        const size_t data_length);


/*!
*   \brief Get the data, dimensions, and type for a tensor in the dataset
*   \details The memory returned in data is valid until the dataset is
*            destroyed. This method is meant to be used when the dimensions
*            and type of the tensor are unknown or the user does not want to
*            manage memory.  However, given that the memory associated with the
*            return data is valid until dataset destruction, this method should
*            not be used repeatedly for large tensor data.  Instead  it is
*            recommended that the user use unpack_dataset_tensor() for large
*            tensor data and to limit memory use by the dataset.
*   \param dataset The dataset to use for this operation
*   \param name The name for the tensor in the dataset
*   \param name_length The length of the dataset name string,
*                      excluding null terminating character
*   \param data Receives data for the tensor, allocated by the library
*   \param dims Receives the number of elements in each dimension of the tensor
*   \param n_dims Receives the number of dimensions for the tensor
*   \param type Receives the retrieved tensor type
*   \param mem_layout The requested memory layout to which newly allocated
*          memory should conform
*   \return Returns SRNoError on success or an error code on failure
*/
SRError get_dataset_tensor(void* dataset,
                           const char* name,
                           const size_t name_length,
                           void** data,
                           size_t** dims,
                           size_t* n_dims,
                           SRTensorType* type,
                           const SRMemoryLayout mem_layout);

/*!
*   \brief Retrieve tensor data into a caller-provided memory buffer with
*          a specified MemoryLayout. This method is the most
*          memory efficient way to retrieve tensor data from a dataset.
*   \details The provided type and dimensions are checked against retrieved
*            values to ensure the provided memory space is sufficient.
*   \param dataset The dataset to use for this operation
*   \param name The name for the tensor in the dataset
*   \param name_length The length of the dataset name string,
*                      excluding null terminating character
*   \param data The buffer into which to receive tensor data
*   \param dims The number of elements provided in each dimension of
*          the supplied memory buffer
*   \param n_dims The number of dimensions in the supplied memory buffer
*   \param type The tensor type for the supplied memory buffer
*   \param mem_layout The memory layout for the supplied memory buffer
*   \return Returns SRNoError on success or an error code on failure
*/
SRError unpack_dataset_tensor(void* dataset,
                              const char* name,
                              const size_t name_length,
                              void* data,
                              const size_t* dims,
                              const size_t n_dims,
                              const SRTensorType type,
                              const SRMemoryLayout mem_layout);

/*!
*   \brief Retrieve metadata scalar field values from the DataSet.
*          This function will allocate and retain management of the
*          memory for the scalar data. For string scalar metadata,
*          use the get_meta_strings() function
*   \param dataset The dataset to use for this operation
*   \param name The name for the metadata field in the DataSet
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param length Receives the number of values returned in \p scalar_data
*   \param type Receives the data type for the metadata field
*   \param scalar_data Receives an array of the metadata field values
*   \return Returns SRNoError on success or an error code on failure
*/
SRError get_meta_scalars(void* dataset,
                         const char* name,
                         const size_t name_length,
                         size_t* length,
                         SRMetaDataType* type,
                         void** scalar_data);


/*!
*   \brief Retrieve metadata string field values from the dataset.
*          This function will allocate and retain management of the
*          memory for the scalar string data.
*          object and remain valid until the DataSet object is destroyed.
*   \param dataset The dataset to use for this operation
*   \param name The name for the metadata field in the DataSet
*   \param name_length The length of the name string,
*                      excluding null terminating character
*   \param data Receives an array of string values
*   \param n_strings Receives the number of strings returned in \p data
*   \param lengths Receives an array containing the lengths of the strings
*                  returned in \p data
*   \return Returns SRNoError on success or an error code on failure
*/
SRError get_meta_strings(void* dataset,
                         const char* name,
                         const size_t name_length,
                         char*** data,
                         size_t* n_strings,
                         size_t** lengths);

/*!
*   \brief Retrieve the names of tensors in the DataSet
*   \param dataset The dataset to use for this operation
*   \param data Receives an array of tensor names
*   \param n_strings Receives the number of strings returned in \p data
*   \param lengths Receives an array containing the lengths of the strings
*                  returned in \p data
*   \return Returns SRNoError on success or an error code on failure
*/
SRError get_tensor_names(
    void* dataset, char*** data, size_t* n_strings, size_t** lengths);

/*!
*   \brief Retrieve the data type of a Tensor in the DataSet
*   \param dataset The dataset to use for this operation
*   \param name The name of the tensor (null-terminated string)
*   \param name_len The length in bytes of the tensor name
*   \param ttype Receives the type for the specified tensor
*   \return Returns SRNoError on success or an error code on failure
*/
SRError get_tensor_type(
    void* dataset, const char* name, size_t name_len, SRTensorType* ttype);

/*!
*   \brief Retrieve the names of all metadata fields in the DataSet
*   \param dataset The dataset to use for this operation
*   \param data Receives an array of metadata field names
*   \param n_strings Receives the number of strings returned in \p data
*   \param lengths Receives an array containing the lengths of the strings
*                  returned in \p data
*   \return Returns SRNoError on success or an error code on failure
*/
SRError get_metadata_field_names(
    void* dataset, char*** data, size_t* n_strings, size_t** lengths);

/*!
*   \brief Retrieve the data type of a metadata field in the DataSet
*   \param dataset The dataset to use for this operation
*   \param name The name of the metadata field (null-terminated string)
*   \param name_len The length in bytes of the metadata field name
*   \param mdtype Receives the type for the specified metadata field
*   \return Returns SRNoError on success or an error code on failure
*/
 SRError get_metadata_field_type(
    void* dataset, const char* name, size_t name_len, SRMetaDataType* mdtype);

#ifdef __cplusplus
}
#endif
#endif // SMARTREDIS_C_DATASET_H
