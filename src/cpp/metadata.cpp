#include "metadata.h"

MetaData::MetaData()
{
}

MetaData::MetaData(MetaData&& metadata)
{
    /* This is the move constructor for the MetaData object
    */
    this->_meta_msg_map = std::move(metadata._meta_msg_map);
    metadata._meta_msg_map.clear();
    this->_char_array_mem_mgr = std::move(metadata._char_array_mem_mgr);
    this->_char_mem_mgr = std::move(metadata._char_mem_mgr);
    this->_double_mem_mgr = std::move(metadata._double_mem_mgr);
    this->_float_mem_mgr = std::move(metadata._float_mem_mgr);
    this->_int64_mem_mgr = std::move(metadata._int64_mem_mgr);
    this->_uint64_mem_mgr = std::move(metadata._uint64_mem_mgr);
    this->_int32_mem_mgr = std::move(metadata._int32_mem_mgr);
    this->_uint32_mem_mgr = std::move(metadata._uint32_mem_mgr);
    this->_buf = std::move(metadata._buf);
    this->_pb_metadata_msg = std::move(metadata._pb_metadata_msg);
}

MetaData& MetaData::operator=(MetaData&& metadata)
{
    /* This is the move assignment operator.
    */
    if(this!=&metadata) {
        this->_meta_msg_map = std::move(metadata._meta_msg_map);
        metadata._meta_msg_map.clear();
        this->_char_array_mem_mgr = std::move(metadata._char_array_mem_mgr);
        this->_char_mem_mgr = std::move(metadata._char_mem_mgr);
        this->_double_mem_mgr = std::move(metadata._double_mem_mgr);
        this->_float_mem_mgr = std::move(metadata._float_mem_mgr);
        this->_int64_mem_mgr = std::move(metadata._int64_mem_mgr);
        this->_uint64_mem_mgr = std::move(metadata._uint64_mem_mgr);
        this->_int32_mem_mgr = std::move(metadata._int32_mem_mgr);
        this->_uint32_mem_mgr = std::move(metadata._uint32_mem_mgr);
        this->_buf = std::move(metadata._buf);
        this->_pb_metadata_msg = std::move(metadata._pb_metadata_msg);
    }
    return *this;
}

void MetaData::fill_from_buffer(const char* buf, unsigned long long buf_size)
{
    /* This function initializes the top level protobuf message
    with the buffer.  The repeated field metadata messages
    in the top level message are then unpacked (i.e pointers
    to each metadata protobuf message are placed in the
    message map).
    */

    //TODO there doesn't seem to be a way around casting the buf
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

void MetaData::add_value(const std::string& field_name,
                         const std::string& type,
                         const void* value)
{
    /* This functions adds a meta data value to the metadata
    field given by fieldname.  The default behavior is to append
    the field if it exists, and if it does not exist, create the
    field.
    */
    if(!(this->_field_exists(field_name)))
        this->_create_message(field_name, type);

    gpb::Message* msg = this->_meta_msg_map[field_name];
    const gpb::Reflection* refl = msg->GetReflection();
    const gpb::FieldDescriptor* field =
        msg->GetDescriptor()->FindFieldByName("data");

    //TODO add try catch for protobuf errors
    int data_type = this->_get_type_integer(type);
    switch(data_type) {
        case string_type :
            refl->AddString(msg, field, std::string((char*)value));
            break;
        case double_type :
            refl->AddDouble(msg, field, *((double*)value));
            break;
        case float_type :
            refl->AddFloat(msg, field, *((float*)value));
            break;
        case int64_type :
            refl->AddInt64(msg, field, *((int64_t*)value));
            break;
        case uint64_type :
            refl->AddUInt64(msg, field, *((uint64_t*)value));
            break;
        case int32_type :
            refl->AddInt32(msg, field, *((int32_t*)value));
            break;
        case uint32_type :
            refl->AddUInt32(msg, field, *((uint32_t*)value));
            break;
    }
    return;
}

void MetaData::get_values(const std::string& name,
                          const std::string& type,
                          void*& data, size_t& length)
{
    /* This function allocates the memory to return a pointer
    (via pointer reference "data") to the user and sets the value
    of "length" to the number of values in data.
    */

    if(!this->_field_exists(name))
        throw std::runtime_error("The metadata string field "
                                 + name + " does not exist.");

    gpb::Message* msg = this->_meta_msg_map[name];

    int data_type = this->_get_type_integer(type);
    switch(data_type) {
        case string_type :
            this->_get_string_field_values(msg, data, length);
            break;
        case double_type :
            this->_get_numeric_field_values<double>
                (msg, data, length, this->_double_mem_mgr);
            break;
        case float_type :
            this->_get_numeric_field_values<float>
                (msg, data, length, this->_float_mem_mgr);
            break;
        case int64_type :
            this->_get_numeric_field_values<int64_t>
                (msg, data, length, this->_int64_mem_mgr);
            break;
        case uint64_type :
            this->_get_numeric_field_values<uint64_t>
                (msg, data, length, this->_uint64_mem_mgr);
            break;
        case int32_type :
            this->_get_numeric_field_values<int32_t>
                (msg, data, length, this->_int32_mem_mgr);
            break;
        case uint32_type :
            this->_get_numeric_field_values<uint32_t>
                (msg, data, length, this->_uint32_mem_mgr);
            break;
    }
    return;
}

std::string_view MetaData::get_metadata_buf()
{
    /* This function will return a string_view of the
    protobuf metadata messages that have been constructed.
    */
    this->_pb_metadata_msg.SerializeToString(&(this->_buf));
    return std::string_view(this->_buf.c_str(), this->_buf.length());
}

void MetaData::_unpack_metadata(const std::string& field_name)
{
    /* This function unpacks the top level meta data
    repeated fields that contain protobuf messages of
    the metadata values.
    */
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
                               const std::string& type)
{
    /* This function will create a new protobuf message to
    hold the metadata of the specified type.
    */
    int data_type = this->_get_type_integer(type);
    switch(data_type) {
        case string_type :
            this->_create_message_by_type
                    <StringMsg>(field_name,top_string_msg);
            break;
        case double_type :
            this->_create_message_by_type
                    <DoubleMsg>(field_name,top_double_msg);
            break;
        case float_type :
            this->_create_message_by_type
                    <FloatMsg>(field_name,top_float_msg);
            break;
        case int64_type :
            this->_create_message_by_type
                    <Int64Msg>(field_name,top_int64_msg);
            break;
        case uint64_type :
            this->_create_message_by_type
                    <UInt64Msg>(field_name,top_uint64_msg);
            break;
        case int32_type :
            this->_create_message_by_type
                    <Int32Msg>(field_name,top_int32_msg);
            break;
        case uint32_type :
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
    /* This funcation creates a protobuf message for the
    metadata field, adds the protobuf message to the map
    holding all metadata messages, and sets the internal
    id field of the message to field_name.
    */

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
    /* This function sets a string in a string message field
    */
    //TODO add check to make sure the field is a string field
    const gpb::Reflection* refl = msg->GetReflection();
    const gpb::FieldDescriptor* field_desc =
        msg->GetDescriptor()->FindFieldByName(field_name);
    refl->SetString(msg, field_desc, value);
    return;
}


void MetaData::_get_string_field_values(gpb::Message* msg,
                                        void*& data,
                                        size_t& n_values) {
    /* This funcition retrieves all of the metadata
    "data" string fields from the message.  A copy
    of the protobuf values is made using the
    MemoryList objects in the MetaData class,
    and as a result, the metadata values passed back
    to the user will live as long as the MetaData object
    persists.
    */
    const gpb::Reflection* refl = msg->GetReflection();
    const gpb::FieldDescriptor* field_desc =
        msg->GetDescriptor()->FindFieldByName("data");

    //Get the number of entries and bytes
    n_values = refl->FieldSize(*msg,field_desc);
    data = (void*)this->_char_array_mem_mgr.allocate(n_values);

    //Copy each metadata string into new char*
    for(int i = 0; i < n_values; i++) {
        std::string value =
            refl->GetRepeatedString(*msg, field_desc, i);
        unsigned size = value.size();
        char* c = this->_char_mem_mgr.allocate(size+1);
        value.copy(c, size, 0);
        c[size]=0;
        ((char**)data)[i] = c;
    }
    return;
}

template <typename T>
void MetaData::_get_numeric_field_values(gpb::Message* msg,
                                         void*& data,
                                         size_t& n_values,
                                         MemoryList<T>& mem_list)
{
    /* This funcition retrieves all of the metadata
    "data" non-string fields from the message.  A copy
    of the protobuf values is made using the
    MemoryList objects in the MetaData class,
    and as a result, the metadata values passed back
    to the user will live as long as the MetaData object
    persists.
    */
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
    /* Return true if the field name is already in use
    for a protobuf messsage, otherwise false.
    */
   return (_meta_msg_map.count(field_name)>0);
}

inline int MetaData::_get_type_integer(const std::string& type)
{
    /* Returns the integer type value corresponding
    to the provided type.  This function will
    throw an error if the type is not valid.
    Error checking is not separated from the function
    because it is implicit in the unordered_map.at()
    */
    try {
        return meta_type_map.at(type);
    }
    catch (const std::out_of_range& oor) {
        throw std::runtime_error("The provide type is invalid: " + type);
    }
}