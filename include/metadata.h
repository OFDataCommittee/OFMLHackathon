#ifndef SMARTSIM_METADATA_H
#define SMARTSIM_METADATA_H

#include "stdlib.h"
#include <string>
#include <unordered_map>
#include "silc.pb.h"
#include <google/protobuf/reflection.h>
#include <google/protobuf/stubs/port.h>
#include "memorylist.h"

namespace gpb = google::protobuf;
namespace spb = SILCProtobuf;

//Typedef protobuf message names in case proto file changes
typedef spb::RepeatedStringMeta StringMsg;
typedef spb::RepeatedDoubleMeta DoubleMsg;
typedef spb::RepeatedFloatMeta FloatMsg;
typedef spb::RepeatedSInt64Meta Int64Msg;
typedef spb::RepeatedUInt64Meta UInt64Msg;
typedef spb::RepeatedSInt32Meta Int32Msg;
typedef spb::RepeatedUInt32Meta UInt32Msg;

//An unordered_map of std::string type (key) and value
//integer that to cut down on strcmp throughout the code
static const int string_type = 1;
static const int double_type = 2;
static const int float_type = 3;
static const int int64_type = 4;
static const int uint64_type = 5;
static const int int32_type = 6;
static const int uint32_type = 7;

static const std::unordered_map<std::string, int>
    meta_type_map{ {"STRING", string_type},
                   {"DOUBLE", double_type},
                   {"FLOAT", float_type},
                   {"INT64", int64_type},
                   {"UINT64", uint64_type},
                   {"INT32", int32_type},
                   {"UINT32", uint32_type} };

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

///@file
///\brief The Command class for constructing meta data messages
class MetaData;

class MetaData
{
    public:

        //! MetaData constructor
        MetaData();

        //! MetaData move constructor
        MetaData(MetaData&& metadata);

        //! MetaData move assignment operator
        MetaData& operator=(MetaData&& metadata);

        //! Reconstruct metadata from buffer
        void fill_from_buffer(const char* buf /*!< The buffer used to fill the metadata object*/,
                              unsigned long long buf_size /*!< The length of the buffer*/
                              );

        //! Add a value to a metadata field
        void add_value(const std::string& field_name /*!< The name of the metadata field*/,
                       const std::string& type /*!< The data type of the field*/,
                       const void* value /*!< The value of the metadata field*/
                       );

        //! Get metadata values from field
        void get_values(const std::string& name /*!< The name of the metadata field*/,
                        std::string& type /*!< The data type of the field*/,
                        void*& data /*!< The pointer that will be pointed to the metadata*/,
                        size_t& length /*!< An integer that will be set to the number of values in the metadata field*/
                        );

        //! Get the metadata fields as a buffer
        std::string_view get_metadata_buf();

    private:

        //! The protobuf message holding all metadata
        spb::MetaData _pb_metadata_msg;

        //! Maps meta data field name to the protobuf metadata message
        std::unordered_map<std::string, gpb::Message*> _meta_msg_map;

        //! MemoryList objects for each metadata type for managing memalloc
        MemoryList<char*>_char_array_mem_mgr;
        MemoryList<char> _char_mem_mgr;
        MemoryList<double> _double_mem_mgr;
        MemoryList<float> _float_mem_mgr;
        MemoryList<int64_t> _int64_mem_mgr;
        MemoryList<uint64_t> _uint64_mem_mgr;
        MemoryList<int32_t> _int32_mem_mgr;
        MemoryList<uint32_t> _uint32_mem_mgr;

        //! The metadata buffer
        std::string _buf;

        //! Unpacks the meta data repeated field in the top
        //! level message and puts message pointers into _meta_msg_map
        void _unpack_metadata(const std::string& field_name /*!< The name of the metadata field*/
                              );

        //! Create a metadata message for the field and type
        void _create_message(const std::string& field_name /*!< The name of the metadata field*/,
                             const std::string& type /*!< The data type of the field*/
                             );

        //! Templated function for creating metadata field
        template <typename PB>
        void _create_message_by_type(const std::string& field_name /*!< The name of the metadata field*/,
                                     const std::string& container_name /*!< The name of the top level message container*/
                                     );


        //! Set a string value to a string field (non-repeated field)
        void _set_string_in_message(gpb::Message* msg /*!< Protobuf message containing the string field*/,
                                    const std::string& field_name /*!< The name of the metadata field*/,
                                    std::string value /*!< The field value*/
                                    );

        //! Get an array of string metadata values
        void _get_string_field_values(gpb::Message* msg /*!< Protobuf message containing the string field*/,
                                     void*& data /*!< The pointer that will be pointed to the metadata*/,
                                     size_t& length /*!< An integer that will be set to the number of values in the metadata field*/
                                     );

        //! Get non-string numeric field values
        template <typename T>
        void _get_numeric_field_values(gpb::Message* msg /*!< Protobuf message containing the string field*/,
                                       void*& data /*!< The pointer that will be pointed to the metadata*/,
                                       size_t& n_values /*!< An integer that will be set to the number of values in the metadata field*/,
                                       MemoryList<T>& mem_list /*!< Memory manager for the metadata heap allocations*/
                                       );

        //! Check if a metadata field already exists by the name
        bool _field_exists(const std::string& field_name /*!< The name of the metadata field*/
                           );

        //! Gets the type integer
        int _get_type_integer(const std::string& type /*!< The data type of the metadata field*/
                              );

        //! Get the type of the metadata value by name.
        std::string _get_meta_value_type(const std::string& name);
};

#endif //SMARTSIM_METADATA_H