#ifndef SMARTREDIS_METADATABUFFER_H
#define SMARTREDIS_METADATABUFFER_H

#include <iostream>
#include <vector>
#include <string>

using namespace SmartRedis;

namespace MetadataBuffer {

typedef int8_t type_t;

extern inline MetaDataType get_type(const std::string_view& buf)
{
    if(buf.size() < sizeof(type_t))
        throw std::runtime_error("The MetadataField type cannot "\
                                 "be retrived from buffer of " +
                                 std::to_string(buf.size()) +
                                 "characters.");

    void* data = (void*)(buf.data());
    int8_t type = *((int8_t*)data);
    return (MetaDataType)type;
}

extern inline void add_buf_data(std::string& buf,
                                          size_t pos,
                                          const void* data,
                                          size_t n_bytes)
{
    std::memcpy(buf.data()+pos, data, n_bytes);
    return;
}

template <typename T>
extern inline bool safe_to_read(const size_t& byte_position,
                                          const size_t& total_bytes,
                                          const size_t& n_values)
{

    if( (total_bytes - byte_position)/sizeof(T) < n_values )
        throw std::runtime_error("An attempt was made to read "\
                                 "a value beyond the "\
                                 "metadata field buffer.  "\
                                 "The buffer may be corrupted.");

    return true;
}

template <typename T>
extern inline T read(void* buf,
                               const size_t& byte_position,
                               const size_t& total_bytes,
                               const size_t& n_values)
{
    safe_to_read<T>(byte_position, total_bytes, n_values);
    return *((T*)buf);
}

template <typename T>
extern inline bool advance(void*& buf,
                                     size_t& byte_position,
                                     const size_t& total_bytes,
                                     const size_t& n_values)
{
    // TODO Check overflow error of additions
    size_t final_byte_position = byte_position +
                                 n_values * sizeof(T);
    if(final_byte_position >= total_bytes)
        return false;
    byte_position = final_byte_position;
    buf = (T*)buf + n_values;
    return true;
}

extern inline std::string read_string(void* buf,
                                                const size_t& byte_position,
                                                const size_t& total_bytes,
                                                const size_t& n_chars)
{
    safe_to_read<char>(byte_position, total_bytes, n_chars);
    return std::string((char*)buf, n_chars);
}


template <typename T>
extern inline std::string generate_scalar_buf(MetaDataType type,
                                              const std::vector<T>& data)
{
    /*
    *   The serialized string has the format described
    *   below.
    *
    *   Field           Field size
    *   -------         ----------
    *   type_id         1 byte
    *
    *   n_values        sizeof(size_t)
    *
    *   data content     sizeof(T) * values.size()
    */

    if(sizeof(int8_t) != sizeof(char))
        throw std::runtime_error("Metadata is not supported on "\
                                 "systems with char length not "\
                                 "equal to 8 bits.");

    // Number of bytes needed for the type identifier
    size_t type_bytes = sizeof(type_t);

    // Number of bytes needed for the scalar values
    size_t data_bytes = sizeof(T) * data.size();

    size_t n_bytes = type_bytes + data_bytes;

    std::string buf(n_bytes, 0);
    size_t pos = 0;

    // Add the type ID
    type_t type_id = (type_t)type;
    n_bytes = sizeof(type_t);
    add_buf_data(buf, pos, &type_id, n_bytes);
    pos += n_bytes;

    // Add the values
    n_bytes = sizeof(T) * data.size();
    const T* v_data = data.data();
    add_buf_data(buf, pos, v_data, n_bytes);
    return buf;
}

extern inline std::string generate_string_buf(
    const std::vector<std::string>& data)
{
    /*
    *   The serialized string has the format described
    *   below.
    *
    *   Field           Field size
    *   -------         ----------
    *   type_id         1 byte
    *
    *   str length      sizeof(size_t)
    *   str content     sizeof(char) * str[i].size()
    *   .               .
    *   .               .
    *   str length      sizeof(size_t)
    *   str content     sizeof(char) * str[i].size()
    */

    // Number of bytes needed for the type identifier
    size_t type_bytes = sizeof(type_t);
    // Number of bytes needed for the string lengths
    size_t str_length_bytes = sizeof(size_t) * data.size();
    // Number of bytes needed for the string values
    size_t data_bytes = 0;
    for(size_t i=0; i<data.size(); i++)
        data_bytes += data[i].size();

    size_t n_bytes = type_bytes + str_length_bytes +
                     data_bytes;

    std::string buf(n_bytes, 0);

    size_t pos = 0;

    // Add the type ID
    type_t type_id = (type_t)MetaDataType::string;
    n_bytes = sizeof(type_t);
    add_buf_data(buf, pos, &type_id, n_bytes);
    pos += n_bytes;

    // Add each string length and string value
    size_t str_length;
    size_t entry_bytes;
    for(size_t i=0; i<data.size(); i++) {
        str_length = data[i].size();
        entry_bytes = sizeof(size_t);
        add_buf_data(buf, pos, &str_length, entry_bytes);
        pos += entry_bytes;

        add_buf_data(buf, pos,
                                      (void*)(data[i].data()),
                                      data[i].size());
        pos += data[i].size();
    }
    return buf;
}

extern inline std::vector<std::string> unpack_string_buf(
    const std::string_view& buf)
{
    size_t byte_position = 0;
    size_t total_bytes = buf.size();

    if(total_bytes==0)
        return std::vector<std::string>();

    void* data = (void*)(buf.data());

    type_t type = read<type_t>(data,
                                                byte_position,
                                                total_bytes, 1);


    if(type!=(type_t)MetaDataType::string)
        throw std::runtime_error("The buffer string metadata type "\
                                 "does not contain the expected "\
                                 "type of string.");

    std::vector<std::string> vals;


    if(!advance<type_t>(data, byte_position,
                                         total_bytes, 1))
        return vals;

    size_t str_len;

    while(byte_position < total_bytes) {

        str_len = read<size_t>(data, byte_position,
                                                total_bytes, 1);

        if(!advance<size_t>(data, byte_position,
                                             total_bytes, 1))
            return vals;

        vals.push_back(
            read_string(data, byte_position,
                                         total_bytes, str_len));

        if(!advance<char>(data, byte_position,
                                           total_bytes, str_len))
            return vals;
    }
    return vals;
}

template <class T>
extern inline std::vector<T> unpack_scalar_buf(const std::string_view& buf)
{
    void* data = (void*)(buf.data());

    size_t byte_position = 0;
    size_t total_bytes = buf.size();

    type_t type = read<type_t>(data,
                                byte_position,
                                total_bytes, 1);


    if(!advance<type_t>(data, byte_position,
                         total_bytes, 1))
        return;

    if( (total_bytes - byte_position) % sizeof(T))
        throw std::runtime_error("The data portion of the provided "\
                                 "metadata buffer does not contain "
                                 "the correct number of bytes for "
                                 "a " + std::to_string(sizeof(T)) +
                                 " byte scalar type. It contains " +
                                 std::to_string(total_bytes -
                                 byte_position) + " bytes");

    /*
    if(type!=(int8_t)this->type())
        throw std::runtime_error("The buffer scalar metadata type "\
                                 "does not match the object type "\
                                 "being used to interpret it.");
    */

    size_t n_vals = (total_bytes - byte_position) / sizeof(T);
    std::vector<T> vals(n_vals);

    std::memcpy(vals.data(), (T*)data, total_bytes - byte_position);
    return vals;
}

} //namespace MetadataBuffer

#endif