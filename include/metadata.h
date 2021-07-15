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

#ifndef SMARTREDIS_METADATA_H
#define SMARTREDIS_METADATA_H

#include "stdlib.h"
#include <string>
#include <vector>
#include <unordered_map>
#include "sharedmemorylist.h"
#include "enums/cpp_metadata_type.h"
#include "metadatafield.h"
#include "metadatabuffer.h"
#include "scalarfield.h"
#include "stringfield.h"

///@file

namespace SmartRedis {

/* These string and map are used by the Python client
to avoid using enums.  This may need to change.
*/
static std::string DATATYPE_METADATA_STR_DOUBLE = "DOUBLE";
static std::string DATATYPE_METADATA_STR_FLOAT = "FLOAT";
static std::string DATATYPE_METADATA_STR_INT32 = "INT32";
static std::string DATATYPE_METADATA_STR_INT64 = "INT64";
static std::string DATATYPE_METADATA_STR_UINT32 = "UINT32";
static std::string DATATYPE_METADATA_STR_UINT64 = "UINT64";
static std::string DATATYPE_METADATA_STR_STRING = "STRING";

static const std::unordered_map<std::string, MetaDataType>
    METADATA_TYPE_MAP{
        {DATATYPE_METADATA_STR_DOUBLE, MetaDataType::dbl},
        {DATATYPE_METADATA_STR_FLOAT, MetaDataType::flt},
        {DATATYPE_METADATA_STR_INT32, MetaDataType::int32},
        {DATATYPE_METADATA_STR_INT64, MetaDataType::int64},
        {DATATYPE_METADATA_STR_UINT32, MetaDataType::uint32},
        {DATATYPE_METADATA_STR_UINT64, MetaDataType::uint64},
        {DATATYPE_METADATA_STR_STRING, MetaDataType::string} };

class MetaData;

/*!
*   \brief The MetaData class stages metadata fields and
           values.  Memory associated with metadata
           retrieval from the MetaData object is valid
           until the MetaData object is destroyed.
*/
class MetaData
{
    public:

        /*!
        *   \brief Default MetaData constructor
        */
        MetaData() = default;

        /*!
        *   \brief MetaData copy constructor
        *   \param metadata The MetaData object
        *                   to copy for construction
        */
        MetaData(const MetaData& metadata);

        /*!
        *   \brief MetaData default move constructor
        *   \param metadata The MetaData object
        *                   to move for construction
        */
        MetaData(MetaData&& metadata) = default;

        /*!
        *   \brief MetaData copy assignment operator
        *   \param metadata The MetaData object
        *                   to copy for assignment
        *   \returns MetaData object reference that has
        *            been assigned the values
        */
        MetaData& operator=(const MetaData&);

        /*!
        *   \brief MetaData move assignment operator
        *   \param metadata The MetaData object
        *                   to move for assignment
        *   \returns MetaData object reference that has
        *            been assigned the values
        */
        MetaData& operator=(MetaData&& metadata);

        /*!
        *   \brief Metadata destructor
        */
        ~MetaData();

        /*!
        *   \brief Add metadata scalar field (non-string) with value.
        *          If the field does not exist, it will be created.
        *          If the field exists, the value
        *          will be appended to existing field.
        *   \param field_name The name of the field in which to
        *                     place the value
        *   \param value A pointer to the value
        *   \param type The MetaDataType of the value
        */
        void add_scalar(const std::string& field_name,
                        const void* value,
                        const MetaDataType type);

        /*!
        *   \brief Add string to a metadata field.
        *          If the field does not exist, it will be created.
        *          If the field exists, the value
        *          will be appended to existing field.
        *   \param field_name The name of the field in which to
        *                     place the value
        *   \param value The string value to add to the field
        */
        void add_string(const std::string& field_name,
                        const std::string& value);

        /*!
        *   \brief Add a serialized field to the MetaData object
        *   \param name The name of the field
        *   \param buf The buffer used for field construction
        *   \param buf_size The length of the buffer
        */
        void add_serialized_field(const std::string& name,
                                  char* buf,
                                  size_t buf_size);

        /*!
        *   \brief  Get metadata values from field
        *           that are scalars (non-string)
        *   \details This function allocates memory to
        *            return a pointer (via pointer reference "data")
        *            to the user and sets the value of "length" to
        *            the number of values in data.  The MetaDataType
        *            reference is also set to the type of the metadata
        *            field because it is assumed this user is unaware
        *            of type.  The allocated memory is valid
        *            until the MetaData object is destroyed.
        *   \param name The name of the field to retrieve
        *   \param data A c-ptr pointed to newly allocated memory
        *               for the metadata field.
        *   \param length The number of elements in the field
        *   \param type The MetaDataType of the retrieved field
        */
        void get_scalar_values(const std::string& name,
                               void*& data,
                               size_t& length,
                               MetaDataType& type);

        /*!
        *   \brief  Get metadata values string field
        *   \details The string field is returned as a std::vector
        *            of std::string which means that the memory
        *            for the field is managed by the returned object.
        *   \param name The name of the string field to retrieve
        *   \returns A vector of the strings in the field
        */
        std::vector<std::string> get_string_values(const std::string& name);

        /*!
        *   \brief  Get metadata string field using a c-style
        *           interface.
        *   \details This function allocates memory to
        *            return a pointer (via pointer reference "data")
        *            to the user and sets the value of n_strings to
        *            the number of strings in the field.  Memory is also
        *            allocated to store the length of each string in the
        *            field, and the provided lengths pointer is pointed
        *            to this new memory.  The memory for the strings and
        *            string lengths is valid until the MetaData object is
        *            destroyed.
        *   \param name The name of the field to retrieve
        *   \param data A c-ptr pointed to newly allocated memory
        *               for the metadata field.
        *   \param n_strings The number of strings in the field
        *   \param lengths A size_t pointer pointed to newly allocated
        *                  memory that stores the length of each string
        */
        void get_string_values(const std::string& name,
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
        *   \brief Returns a vector of std::pair with
        *          the field name and the field serialization
        *          for all fields in the MetaData set.
        *   \returns std::pair<std::string, std::string> containing
        *            the field name and the field serialization.
        */
        std::vector<std::pair<std::string, std::string>>
            get_metadata_serialization_map();


    private:


        /*!
        *   \brief Maps metadata field name to the
        *          MetaDataField object.
        */
        std::unordered_map<std::string, MetadataField*> _field_map;

        /*!
        *   \brief SharedMemoryList for arrays of c-str
        *          memory allocation associated with retrieving
        *          metadata
        */
        SharedMemoryList<char*>_char_array_mem_mgr;

        /*!
        *   \brief SharedMemoryList for c-str memory
        *          allocation associated with retrieving metadata
        */
        SharedMemoryList<char> _char_mem_mgr;

        /*!
        *   \brief SharedMemoryList for double memory
        *          allocation associated with retrieving metadata
        */
        SharedMemoryList<double> _double_mem_mgr;

        /*!
        *   \brief SharedMemoryList for float memory
        *          allocation associated with retrieving metadata
        */
        SharedMemoryList<float> _float_mem_mgr;

        /*!
        *   \brief SharedMemoryList for int64 memory
        *          allocation associated with retrieving metadata
        */
        SharedMemoryList<int64_t> _int64_mem_mgr;

        /*!
        *   \brief SharedMemoryList for uint64 memory
        *          allocation associated with retrieving metadata
        */
        SharedMemoryList<uint64_t> _uint64_mem_mgr;

        /*!
        *   \brief SharedMemoryList for int32 memory
        *          allocation associated with retrieving metadata
        */
        SharedMemoryList<int32_t> _int32_mem_mgr;

        /*!
        *   \brief SharedMemoryList for uint32 memory
        *          allocation associated with retrieving metadata
        */
        SharedMemoryList<uint32_t> _uint32_mem_mgr;

        /*!
        *   \brief SharedMemoryList for size_t memory
        *          allocation associated with retrieving
        *          string field sting lengths
        */
        SharedMemoryList<size_t> _str_len_mem_mgr;

        /*!
        *   \brief Create a new metadata field with the given
        *          name and type.
        *   \param field_name The name of the metadata field
        *   \param type The data type of the field
        */
        void _create_field(const std::string& field_name,
                           const MetaDataType type);

        /*!
        *   \brief This function will perform a deep copy assignment
        *          of a scalar or string field.
        *   \param dest_field The destination field
        *   \param src_field The source field
        */
        void _deep_copy_field(MetadataField* dest_field,
                              MetadataField* src_field);


        /*!
        *   \brief This function creates a new scalar metadata field
        *          and adds it to the field map.
        *   \tparam The datatype associated with the MetaDataType
        *           (e.g. double)
        *   \param field_name The name of the metadata field
        *   \param type The type of field (e.g. double, float, string)
        */
        template <typename T>
        void _create_scalar_field(const std::string& field_name,
                                  const MetaDataType type);

        /*!
        *   \brief This function creates a new string metadata field
        *          and adds it to the field map.
        *   \param field_name The name of the metadata field
        */
        void _create_string_field(const std::string& field_name);

        /*!
        *   \brief This function allocates new memory to hold
        *          metadata field values and returns these values
        *          via the c-ptr reference being pointed to the
        *          newly allocated memory.
        *   \tparam The type associated with the SharedMemoryList
        *   \param data A c-ptr reference where metadata values
        *               will be placed after allocation of memory
        *   \param n_values A reference to a variable holding the
        *                   number of values in the field
        *   \param mem_list A reference to a SharedMemoryList that
        *                   is used to allocate new memory for the
        *                   return metadata field values.
        */
        template <typename T>
        void _get_numeric_field_values(const std::string& name,
                                       void*& data,
                                       size_t& n_values,
                                       SharedMemoryList<T>& mem_list);

        /*!
        *   \brief Delete the memory associated with all fields
        *          and clear inventory
        */
        void _delete_fields();

};

} //namespace SmartRedis

#endif //SMARTREDIS_METADATA_H
