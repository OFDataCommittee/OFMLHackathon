#include "metadata.h"

using namespace SmartRedis;

MetaData::MetaData(const MetaData& metadata) {
    /* Copy constructor for Metadata
    */
    this->_pb_metadata_msg = metadata._pb_metadata_msg;
    this->_rebuild_message_map();
    this->_char_array_mem_mgr = metadata._char_array_mem_mgr;
    this->_char_mem_mgr = metadata._char_mem_mgr;
    this->_double_mem_mgr = metadata._double_mem_mgr;
    this->_float_mem_mgr = metadata._float_mem_mgr;
    this->_int64_mem_mgr = metadata._int64_mem_mgr;
    this->_uint64_mem_mgr = metadata._uint64_mem_mgr;
    this->_int32_mem_mgr = metadata._int32_mem_mgr;
    this->_uint32_mem_mgr = metadata._uint32_mem_mgr;
    this->_str_len_mem_mgr = metadata._str_len_mem_mgr;
    this->_buf = metadata._buf;
}

MetaData& MetaData::operator=(const MetaData& metadata) {

    if(this!=&metadata) {
        this->_pb_metadata_msg = metadata._pb_metadata_msg;
        this->_rebuild_message_map();
        this->_char_array_mem_mgr = metadata._char_array_mem_mgr;
        this->_char_mem_mgr = metadata._char_mem_mgr;
        this->_double_mem_mgr = metadata._double_mem_mgr;
        this->_float_mem_mgr = metadata._float_mem_mgr;
        this->_int64_mem_mgr = metadata._int64_mem_mgr;
        this->_uint64_mem_mgr = metadata._uint64_mem_mgr;
        this->_int32_mem_mgr = metadata._int32_mem_mgr;
        this->_uint32_mem_mgr = metadata._uint32_mem_mgr;
        this->_str_len_mem_mgr = metadata._str_len_mem_mgr;
        this->_buf = metadata._buf;
    }
    return *this;
}

void MetaData::fill_from_buffer(const char* buf, unsigned long long buf_size)
{
    //TODO There doesn't seem to be a way around casting the buf
    //as a std::string which means that the buffer will be copied
    _pb_metadata_msg.ParseFromString(std::string(buf,buf_size));
    this->_unpack_metadata(top_string_msg);
    this->_unpack_metadata(top_double_msg);
    this->_unpack_metadata(top_float_msg);
    this->_unpack_metadata(top_int64_msg);
    this->_unpack_metadata(top_uint64_msg);
    this->_unpack_metadata(top_int32_msg);
    this->_unpack_metadata(top_uint32_msg);
    return;
}

void MetaData::add_scalar(const std::string& field_name,
                          const void* value,
                          MetaDataType type)
{
    if(!(this->_field_exists(field_name))) {
         this->_create_message(field_name, type);
    }

    gpb::Message* msg = this->_meta_msg_map[field_name];
    const gpb::Reflection* refl = msg->GetReflection();
    const gpb::FieldDescriptor* field =
        msg->GetDescriptor()->FindFieldByName("data");

    //TODO add try catch for protobuf errors
    switch(type) {
        case MetaDataType::string :
            throw std::runtime_error("Use add_string() to add "\
                                     "a string to the metadata "\
                                     "field.");
            break;
        case MetaDataType::dbl :
            refl->AddDouble(msg, field, *((double*)value));
            break;
        case MetaDataType::flt :
            refl->AddFloat(msg, field, *((float*)value));
            break;
        case MetaDataType::int64 :
            refl->AddInt64(msg, field, *((int64_t*)value));
            break;
        case MetaDataType::uint64 :
            refl->AddUInt64(msg, field, *((uint64_t*)value));
            break;
        case MetaDataType::int32 :
            refl->AddInt32(msg, field, *((int32_t*)value));
            break;
        case MetaDataType::uint32 :
            refl->AddUInt32(msg, field, *((uint32_t*)value));
            break;
        default :
            throw std::runtime_error("Unsupported type used "\
                                     "in MetaData.add_scalar().");
            break;
    }
    return;
}

void MetaData::add_string(const std::string& field_name,
                          const std::string& value)
{
    if(!(this->_field_exists(field_name)))
         this->_create_message(field_name, MetaDataType::string);

    gpb::Message* msg = this->_meta_msg_map[field_name];
    const gpb::Reflection* refl = msg->GetReflection();
    const gpb::FieldDescriptor* field =
        msg->GetDescriptor()->FindFieldByName("data");

    refl->AddString(msg, field, value);

    return;
}

void MetaData::get_scalar_values(const std::string& name,
                                void*& data,
                                size_t& length,
                                MetaDataType& type)
{
    if(!this->_field_exists(name))
        throw std::runtime_error("The metadata field "
                                 + name +
                                 " does not exist.");

    type = this->_get_meta_value_type(name);
    gpb::Message* msg = this->_meta_msg_map[name];
    switch(type) {
        case MetaDataType::dbl :
            this->_get_numeric_field_values<double>
                (msg, data, length, this->_double_mem_mgr);
            break;
        case MetaDataType::flt :
            this->_get_numeric_field_values<float>
                (msg, data, length, this->_float_mem_mgr);
            break;
        case MetaDataType::int64 :
            this->_get_numeric_field_values<int64_t>
                (msg, data, length, this->_int64_mem_mgr);
            break;
        case MetaDataType::uint64 :
            this->_get_numeric_field_values<uint64_t>
                (msg, data, length, this->_uint64_mem_mgr);
            break;
        case MetaDataType::int32 :
            this->_get_numeric_field_values<int32_t>
                (msg, data, length, this->_int32_mem_mgr);
            break;
        case MetaDataType::uint32 :
            this->_get_numeric_field_values<uint32_t>
                (msg, data, length, this->_uint32_mem_mgr);
            break;
        case MetaDataType::string :
            throw std::runtime_error("MetaData.get_scalar_values() "\
                                     "cannot retrieve strings.");
            break;
    }
    return;
}

void MetaData::get_string_values(const std::string& name,
                                 char**& data,
                                 size_t& n_strings,
                                 size_t*& lengths)

{
    std::vector<std::string> field_strings =
        this->get_string_values(name);

    //Copy each metadata string into new char*
    n_strings = field_strings.size();
    data = this->_char_array_mem_mgr.allocate(n_strings);
    lengths = this->_str_len_mem_mgr.allocate(n_strings);
    for(int i = 0; i < n_strings; i++) {
        size_t size = field_strings[i].size();
        char* c = this->_char_mem_mgr.allocate(size+1);
        field_strings[i].copy(c, size, 0);
        c[size]=0;
        data[i] = c;
        lengths[i] = size;
    }

    return;
}

std::vector<std::string>
MetaData::get_string_values(const std::string& name)

{
    if(!this->_field_exists(name))
        throw std::runtime_error("The metadata field "
                                 + name +
                                 " does not exist.");

    MetaDataType type = this->_get_meta_value_type(name);

    if(type!=MetaDataType::string)
        throw std::runtime_error("The metadata field " +
                                 name +
                                 " is not a string field.");

    gpb::Message* msg = this->_meta_msg_map[name];

    return this->_get_string_field_values(msg);
}

std::string_view MetaData::get_metadata_buf()
{
    this->_pb_metadata_msg.SerializeToString(&(this->_buf));
    return std::string_view(this->_buf.c_str(), this->_buf.length());
}

void MetaData::clear_field(const std::string& field_name)
{
    if(this->_field_exists(field_name))
        this->_meta_msg_map[field_name]->Clear();
    return;
}

void MetaData::_unpack_metadata(const std::string& field_name)
{
    gpb::Message* top_msg = &(this->_pb_metadata_msg);

    const gpb::Reflection* refl = top_msg->GetReflection();
    const gpb::FieldDescriptor* field = top_msg->GetDescriptor()->
                                        FindFieldByName(field_name);

    int n = refl->FieldSize(*top_msg, field);

    for(int i = 0; i<n; i++) {

        // Fetch the ith meta data field
        gpb::Message* meta_msg =
            refl->MutableRepeatedMessage(top_msg, field, i);
        const gpb::Reflection* meta_refl =
            meta_msg->GetReflection();

        // Get the ID and store the message pointer in the map
        const gpb::FieldDescriptor* id_field =
            meta_msg->GetDescriptor()->FindFieldByName("id");
        std::string id = meta_refl->GetString(*meta_msg, id_field);

        if(_meta_msg_map.count(id)>0) {
            throw std::runtime_error("Could not unpack " + id +
                                     " because it already exists"\
                                     " in the map");
        }

        _meta_msg_map[id] = meta_msg;
    }
    return;
}

void MetaData::_create_message(const std::string& field_name,
                               const MetaDataType type)
{
    switch(type) {
        case MetaDataType::string :
            this->_create_message_by_type
                    <StringMsg>(field_name,top_string_msg);
            break;
        case MetaDataType::dbl :
            this->_create_message_by_type
                    <DoubleMsg>(field_name,top_double_msg);
            break;
        case MetaDataType::flt :
            this->_create_message_by_type
                    <FloatMsg>(field_name,top_float_msg);
            break;
        case MetaDataType::int64 :
            this->_create_message_by_type
                    <Int64Msg>(field_name,top_int64_msg);
            break;
        case MetaDataType::uint64 :
            this->_create_message_by_type
                    <UInt64Msg>(field_name,top_uint64_msg);
            break;
        case MetaDataType::int32 :
            this->_create_message_by_type
                    <Int32Msg>(field_name,top_int32_msg);
            break;
        case MetaDataType::uint32 :
            this->_create_message_by_type
                    <UInt32Msg>(field_name,top_uint32_msg);
            break;
    }
    return;
}

template <typename PB>
void MetaData::_create_message_by_type(const std::string& field_name,
                                       const std::string& container_name)
{
    // Create a new message
    PB* msg = new PB();
    this->_meta_msg_map[field_name] = msg;

    this->_set_string_in_message(msg, "id", field_name);

    // Add the message to our map
    const gpb::Reflection* refl =
            this->_pb_metadata_msg.GetReflection();

    const gpb::FieldDescriptor* meta_desc =
            this->_pb_metadata_msg.GetDescriptor()->
            FindFieldByName(container_name);

    refl->AddAllocatedMessage(&(this->_pb_metadata_msg),
                             meta_desc, msg);
    return;
}

void MetaData::_set_string_in_message(gpb::Message* msg,
                                      const std::string& field_name,
                                      std::string value)
{
    //TODO add check to make sure the field is a string field
    const gpb::Reflection* refl = msg->GetReflection();
    const gpb::FieldDescriptor* field_desc =
        msg->GetDescriptor()->FindFieldByName(field_name);
    refl->SetString(msg, field_desc, value);
    return;
}

std::vector<std::string> MetaData::_get_string_field_values(gpb::Message* msg)
{
    const gpb::Reflection* refl = msg->GetReflection();
    const gpb::FieldDescriptor* field_desc =
        msg->GetDescriptor()->FindFieldByName("data");

    //Get the number of entries and bytes
    int n_values = refl->FieldSize(*msg,field_desc);

    std::vector<std::string> field_strings(n_values);

    for(int i = 0; i < n_values; i++) {
        field_strings[i] =
            refl->GetRepeatedString(*msg, field_desc, i);
    }
    return field_strings;
}

template <typename T>
void MetaData::_get_numeric_field_values(gpb::Message* msg,
                                         void*& data,
                                         size_t& n_values,
                                         SharedMemoryList<T>& mem_list)
{
    const gpb::Reflection* refl = msg->GetReflection();
    const gpb::FieldDescriptor* field_desc =
        msg->GetDescriptor()->FindFieldByName("data");

    //Get the number of entries and number of bytes
    n_values = refl->FieldSize(*msg,field_desc);
    //Allocate the memory
    data = (void*)mem_list.allocate(n_values);

    const google::protobuf::MutableRepeatedFieldRef<T>
        pb_data = refl->GetMutableRepeatedFieldRef<T>(msg, field_desc);

    //Copy each metadata string into numeric value
    for(int i = 0; i < n_values; i++) {
        ((T*)data)[i] = pb_data.Get(i);
    }
    return;
}

bool MetaData::_field_exists(const std::string& field_name)
{
   return (_meta_msg_map.count(field_name)>0);
}

inline MetaDataType MetaData::_get_meta_value_type(const std::string& name)
{
    gpb::Message* msg = this->_meta_msg_map[name];
    const gpb::Reflection* refl = msg->GetReflection();
    const gpb::FieldDescriptor* field_desc =
        msg->GetDescriptor()->FindFieldByName("data");

    MetaDataType type;
    switch(field_desc->cpp_type()) {
        case gpb::FieldDescriptor::CppType::CPPTYPE_INT32 :
            type = MetaDataType::int32;
            break;
        case gpb::FieldDescriptor::CppType::CPPTYPE_INT64 :
            type = MetaDataType::int64;
            break;
        case gpb::FieldDescriptor::CppType::CPPTYPE_UINT32 :
            type = MetaDataType::uint32;
            break;
        case gpb::FieldDescriptor::CppType::CPPTYPE_UINT64 :
            type = MetaDataType::uint64;
            break;
        case gpb::FieldDescriptor::CppType::CPPTYPE_FLOAT :
            type = MetaDataType::flt;
            break;
        case gpb::FieldDescriptor::CppType::CPPTYPE_DOUBLE :
            type = MetaDataType::dbl;
            break;
        case gpb::FieldDescriptor::CppType::CPPTYPE_STRING :
            type = MetaDataType::string;
            break;
        default :
            throw std::runtime_error("Unexpected type encountered in"\
                                     "MetaData._get_meta_value_type().");
    }
    return type;
}

void MetaData::_rebuild_message_map()
{
    this->_meta_msg_map.clear();

    const gpb::Reflection* refl = this->_pb_metadata_msg.GetReflection();
    std::vector<const gpb::FieldDescriptor*> field_descriptors;
    refl->ListFields(this->_pb_metadata_msg,&field_descriptors);

    std::vector<const gpb::FieldDescriptor*>::const_iterator it =
        field_descriptors.cbegin();
    std::vector<const gpb::FieldDescriptor*>::const_iterator it_end =
        field_descriptors.cend();

    while(it!=it_end) {
        int field_size = refl->FieldSize(this->_pb_metadata_msg, *it);
        for(int i=0; i<field_size; i++) {
            gpb::Message* sub_msg =
                refl->MutableRepeatedMessage(&(this->_pb_metadata_msg),*it, i);
            const gpb::Reflection* sub_refl = sub_msg->GetReflection();
            const gpb::FieldDescriptor* sub_field =
                sub_msg->GetDescriptor()->FindFieldByName("id");
            std::string name = sub_refl->GetString(*sub_msg, sub_field);
            this->_meta_msg_map[name] = sub_msg;
        }
        it++;
    }
    return;
}