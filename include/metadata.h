#ifndef SMARTSIM_METADATA_H
#define SMARTSIM_METADATA_H

#include "stdlib.h"
#include <string>
#include <unordered_map>
#include "silc.pb.h"
#include <google/protobuf/reflection.h>
#include <google/protobuf/stubs/port.h>
#include "memorylist.h"
#include "enums/cpp_metadata_type.h"

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

        //! Add a value to a metadata field (non-string)
        void add_scalar(const std::string& field_name /*!< The name of the metadata field*/,
                        const void* value /*!< The value of the metadata field*/,
                        const MetaDataType type /*!< The data type of the field*/
                        );

        //! Add a value to a metadata field (non-string)
        void add_string(const std::string& field_name /*!< The name of the metadata field*/,
                        const std::string& value /*!< The value of the metadata field*/
                        );

        //! Get metadata values from field that are scalars (non-string)
        void get_scalar_values(const std::string& name /*!< The name of the metadata field*/,
                               void*& data /*!< The pointer that will be pointed to the metadata*/,
                               size_t& length /*!< An integer that will be set to the number of values in the metadata field*/,
                               MetaDataType& type /*!< The data type of the field*/
                               );

        //! Get metadata values from field that are strings
        std::vector<std::string> get_string_values(const std::string& name /*!< The name of the metadata field*/
                                                   );

        //! Get metadata values from field that are strings using a c-style interface
        void get_string_values(const std::string& name /*!< The name of the metadata field*/,
                               char**& data /*!< The pointer that will be pointed to the metadata*/,
                               size_t& n_strings /*!< An integer that will be set to the number of values in the metadata field*/,
                               size_t*& lengths /*!< An array of string lengths provided to the user for iterating over the c-strings*/
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
        MemoryList<size_t> _str_len_mem_mgr;

        //! The metadata buffer
        std::string _buf;

        //! Unpacks the meta data repeated field in the top
        //! level message and puts message pointers into _meta_msg_map
        void _unpack_metadata(const std::string& field_name /*!< The name of the metadata field*/
                              );

        //! Create a metadata message for the field and type
        void _create_message(const std::string& field_name /*!< The name of the metadata field*/,
                             const MetaDataType type /*!< The data type of the field*/
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
        std::vector<std::string> _get_string_field_values(gpb::Message* msg /*!< Protobuf message containing the string field*/
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

        //! Gets the metadata type for a particular field
        MetaDataType _get_meta_value_type(const std::string& name);
};

#endif //SMARTSIM_METADATA_H