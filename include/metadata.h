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
#include "smartredis.pb.h"
#include <google/protobuf/reflection.h>
#include <google/protobuf/stubs/port.h>
#include "sharedmemorylist.h"
#include "enums/cpp_metadata_type.h"

///@file

namespace gpb = google::protobuf;
namespace spb = SmartRedisProtobuf;

namespace SmartRedis {

//Declare the top level container names in the
//protobuf message so that they are not constant
//strings scattered throughout code
static const char* top_string_msg = "repeated_string_meta";
static const char* top_double_msg = "repeated_double_meta";
static const char* top_float_msg = "repeated_float_meta";
static const char* top_int64_msg = "repeated_sint64_meta";
static const char* top_uint64_msg = "repeated_uint64_meta";
static const char* top_int32_msg = "repeated_sint32_meta";
static const char* top_uint32_msg = "repeated_uint32_meta";

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

static const std::unordered_map<MetaDataType, std::string>
    METADATA_STR_MAP{
        {MetaDataType::dbl, DATATYPE_METADATA_STR_DOUBLE},
        {MetaDataType::flt, DATATYPE_METADATA_STR_FLOAT},
        {MetaDataType::int32, DATATYPE_METADATA_STR_INT32},
        {MetaDataType::int64, DATATYPE_METADATA_STR_INT64},
        {MetaDataType::uint32, DATATYPE_METADATA_STR_UINT32},
        {MetaDataType::uint64, DATATYPE_METADATA_STR_UINT64},
        {MetaDataType::string, DATATYPE_METADATA_STR_STRING} };

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
        MetaData& operator=(MetaData&& metadata) = default;

        /*!
        *   \brief Reconstruct metadata from buffer
        *   \details This function initializes the top level
        *            protobuf message with the buffer.
        *            The repeated field metadata messages
        *            in the top level message are then unpacked
        *            (i.e pointers to each metadata protobuf
        *            message are placed in the message map).
        *   \param buf The buffer used to fill the metadata object
        *   \param buf_size The length of the buffer
        */
        void fill_from_buffer(const char* buf,
                              unsigned long long buf_size);

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
        *   \brief  Get a serialized buffer of the MetaData
        *           fields that can be sent to the database.
        *   \returns A std::string_view of the MetaData contents
        *            serialized
        */
        std::string_view get_metadata_buf();

        /*!
        *   \brief This function clears all entries in a
        *          DataSet field.
        *   \param field_name The name of the field to clear
        */
        void clear_field(const std::string& field_name);

        /*!
        *   \typedef The Protobuf message holding string fields
        */
        typedef spb::RepeatedStringMeta StringMsg;

        /*!
        *   \typedef The Protobuf message holding double fields
        */
        typedef spb::RepeatedDoubleMeta DoubleMsg;

        /*!
        *   \typedef The Protobuf message holding float fields
        */
        typedef spb::RepeatedFloatMeta FloatMsg;

        /*!
        *   \typedef The Protobuf message holding int64 fields
        */
        typedef spb::RepeatedSInt64Meta Int64Msg;

        /*!
        *   \typedef The Protobuf message holding uint64 fields
        */
        typedef spb::RepeatedUInt64Meta UInt64Msg;

        /*!
        *   \typedef The Protobuf message holding int32 fields
        */
        typedef spb::RepeatedSInt32Meta Int32Msg;

        /*!
        *   \typedef The Protobuf message holding uint32 fields
        */
        typedef spb::RepeatedUInt32Meta UInt32Msg;

    private:

        /*!
        *   \brief The protobuf message that holds all fields
        */
        spb::MetaData _pb_metadata_msg;

        /*!
        *   \brief Maps meta data field name to the
        *          protobuf metadata message.
        *   \details This map does not need to manage
        *            memory because all of these
        *            fields will be added to the top
        *            level message and that top level
        *            message will handle memory.
        */
        std::unordered_map<std::string, gpb::Message*> _meta_msg_map;

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
        *   \brief The serialized MetaData buffer
        */
        std::string _buf;

        /*!
        *   \brief Unpacks the all fields for a metadata type
        *          (e.g. string, float, int) that are in the
        *          top level protobuf message and puts pointers
        *          to the sub messages into _meta_msg_map
        *   \param field_name The name of the metadata
        *                     field type message in the top
        *                     level message
        */
        void _unpack_metadata(const std::string& field_name);

        /*!
        *   \brief Create a new metadata message for the field
        *          and type
        *   \param field_name The name of the metadata field
        *   \param type The data type of the field
        */
        void _create_message(const std::string& field_name,
                             const MetaDataType type);


        /*!
        *   \brief This function creates a protobuf message for the
        *          metadata field, adds the protobuf message to the map
        *          holding all metadata messages, and sets the internal
        *          id field of the message to field_name.  In this case,
        *          the top level message is the repeated field that
        *          holds all metadata fields for a specific type.
        *   \tparam PB The protobuf message type to create
        *   \param field_name The name of the metadata field
        *   \param container_name The name of the top level message
        *                         container
        */
        template <typename PB>
        void _create_message_by_type(const std::string& field_name,
                                     const std::string& container_name);

        /*!
        *   \brief Set a string value to a string field
        *          (non-repeated field)
        *   \param msg Protobuf message containing the string field
        *   \param field_name The name of the metadata field
        *   \param value The field value
        */
        void _set_string_in_message(gpb::Message* msg,
                                    const std::string& field_name,
                                    std::string value);

        /*!
        *   \brief Get an array of string metadata values
        *   \param msg Protobuf message containing the string field
        *   \returns The strings in the protobuf message
        */
        std::vector<std::string> _get_string_field_values(gpb::Message* msg);

        /*!
        *   \brief This funcition retrieves all of the metadata
        *          "data" non-string fields from the message.  A copy
        *          of the protobuf values is made using the
        *          MemoryList objects in the MetaData class,
        *          and as a result, the metadata values passed back
        *          to the user will live as long as the MetaData object
        *          persists.
        *   \tparam T the type associated with the metdata (e.g. double, float)
        *   \param msg Protobuf message containing the scalar field
        *   \param data The pointer that will be pointed to the metadata
        *   \param n_Values An size_t variable that will be set to the
        *                   number of values in the metadata field
        *   \param mem_list Memory manager for the metadata heap allocations
        *                   associated with copying fetched metadata
        */
        template <typename T>
        void _get_numeric_field_values(gpb::Message* msg,
                                       void*& data,
                                       size_t& n_values,
                                       SharedMemoryList<T>& mem_list);

        /*!
        *   \brief Return true if the field name is already in use
        *          for a protobuf messsage, otherwise false.
        *   \param field_name The name of the metadata field
        *   \returns True if the field name is already in use,
        *            otherwise false
        */
        bool _field_exists(const std::string& field_name);

        /*!
        *   \brief Gets the metadata type for a particular field
        *   \param name The name of the metadata field
        *   \returns The MetaDataType of the field
        */
        MetaDataType _get_meta_value_type(const std::string& name);

        /*!
        *   \brief Rebuild the message map that maps
        *          field name to protobuf message
        *   \details This function should be invoked if
        *            the location of messages in the map changes.
        *            The message map is rebuilt by looping through
        *            all fields in the main protobuf message.
        */
        void _rebuild_message_map();
};

} //namespace SmartRedis

#endif //SMARTREDIS_METADATA_H
