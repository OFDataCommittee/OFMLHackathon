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

#ifndef SMARTREDIS_METADATABUFFER_H
#define SMARTREDIS_METADATABUFFER_H

#include <iostream>
#include <vector>
#include <string>
#include <cstring>

using namespace SmartRedis;

namespace MetadataBuffer {

/*!
*   \brief The data type associated with the type
*          identifier in the serialized message
*/
typedef int8_t type_t;

/*!
*   \brief Determine the MetadataType in a metadata
*          buffer
*   \param buf The metadata buffer
*   \return The MetaDataType embedded in the buffer
*/
extern inline MetaDataType get_type(const std::string_view& buf)
{
    if(buf.size() < sizeof(type_t))
        throw std::runtime_error("The MetadataField type cannot "\
                                 "be retrived from buffer of " +
                                 std::to_string(buf.size()) +
                                 "characters.");

    type_t* data = (type_t*)(buf.data());
    return (MetaDataType)(*data);
}

/*!
*   \brief Add data to the buffer string
*   \param buf The metadata buffer that has sufficient
*              memory to hold the additional data
*   \param pos The position in the buffer to start
*              adding data
*   \param data A c-ptr to the data
*   \param n_bytes The number of bytes to be added
*                  from data to the buffer
*/
extern inline void add_buf_data(std::string& buf,
                                size_t pos,
                                const void* data,
                                size_t n_bytes)
{
    std::memcpy(buf.data()+pos, data, n_bytes);
    return;
}

/*!
*   \brief Check if it is safe to read a set
*          number of values of type T from the
*          buffer
*   \param byte_position The current position in the buffer
*   \param total_bytes The total bytes in the buffer
*   \param n_values The number of values to read
*   \tparam T The data type that is to be read from buffer
*   \return True if it is safe to read from the buffer,
*           otherwise False
*/
template <typename T>
extern inline bool safe_to_read(const size_t& byte_position,
                                const size_t& total_bytes,
                                const size_t& n_values)
{
    return ((total_bytes - byte_position)/sizeof(T) >= n_values);
}

/*!
*   \brief Read a single value of type T from the buffer
*   \param buf The data buffer
*   \param byte_position The current position in the buffer
*   \param total_bytes The total bytes in the buffer
*   \param n_values The number of values to read
*   \tparam T The data type that is to be read from buffer
*   \return The value read from the buffer
*   \throw std::runtime_error if an attempt is made to read
*          beyond the buffer length.
*/
template <typename T>
extern inline T read(void* buf,
                     const size_t& byte_position,
                     const size_t& total_bytes)
{
    if(!safe_to_read<T>(byte_position, total_bytes, 1))
        throw std::runtime_error("A request to read one scalar value "
                                 "from the metadata buffer "
                                 "was made, but the buffer "
                                 "contains insufficient bytes. "
                                 "The buffer contains " +
                                 std::to_string(total_bytes) +
                                 "bytes and is currently at " +
                                 "position " +
                                 std::to_string(byte_position));
    return *((T*)buf);
}

/*!
*   \brief Advance the position in the buffer corresponding
*          to a number of values of a given type
*   \param buf The data buffer
*   \param byte_position The current position in the buffer
*                        that will be updated
*   \param total_bytes The total bytes in the buffer
*   \param n_values The number of values to advance by
*   \tparam T The data type used to calculate the updated
*           position in the buffer
*   \return True if the buffer is advanced, false if it cannot
*           be advanced
*/
template <typename T>
extern inline bool advance(void*& buf,
                           size_t& byte_position,
                           const size_t& total_bytes,
                           const size_t& n_values)
{
    if(!safe_to_read<T>(byte_position, total_bytes, n_values))
        return false;

    byte_position += n_values * sizeof(T);
    buf = (T*)buf + n_values;
    return true;
}

/*!
*   \brief Read a string of length n_chars from the buffer
*   \param buf The data buffer
*   \param byte_position The current position in the buffer
*   \param total_bytes The total bytes in the buffer
*   \param n_chars The number of characters in the string
*                  to be read
*   \return The string read from the buffer
*   \throw std::runtime_error if an attempt is made to read
*          beyond the buffer length.
*/
extern inline std::string read_string(void* buf,
                                      const size_t& byte_position,
                                      const size_t& total_bytes,
                                      const size_t& n_chars)
{
    if(!safe_to_read<char>(byte_position, total_bytes, n_chars))
        throw std::runtime_error("A request to read a string "
                                 "from the metadata buffer "
                                 "was made, but the buffer "
                                 "contains insufficient bytes. "
                                 "The buffer contains " +
                                 std::to_string(total_bytes) +
                                 "bytes and is currently at " +
                                 "position " +
                                 std::to_string(byte_position));
    return std::string((char*)buf, n_chars);
}

/*!
*   \brief Serialize scalar data into a string buffer
*   \param type The MetaDataType associated with the scalar data
*   \param data The std::vector containing scalar data
*   \tparam The type associated with the data
*   \return std::string of the serialized data
*/
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

/*!
*   \brief Serialize string data into a string buffer
*   \param data The std::vector containing string data
*   \return std::string of the serialized data
*/
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
        add_buf_data(buf, pos, (void*)(data[i].data()),
                     data[i].size());
        pos += data[i].size();
    }
    return buf;
}

/*!
*   \brief Unpack a buffer of string values and return
*          a vector of strings
*   \param buf The data buffer
*   \return std::vector<std::string> of the buffer
*           values
*/
extern inline std::vector<std::string> unpack_string_buf(
    const std::string_view& buf)
{
    size_t byte_position = 0;
    size_t total_bytes = buf.size();

    if(total_bytes==0)
        return std::vector<std::string>();

    void* data = (void*)(buf.data());

    type_t type = read<type_t>(data, byte_position, total_bytes);


    if(type!=(type_t)MetaDataType::string)
        throw std::runtime_error("The buffer string metadata type "\
                                 "does not contain the expected "\
                                 "type of string.");

    std::vector<std::string> vals;

    if(!advance<type_t>(data, byte_position, total_bytes, 1))
        return vals;

    size_t str_len;

    while(byte_position < total_bytes) {

        str_len = read<size_t>(data, byte_position, total_bytes);

        if(!advance<size_t>(data, byte_position, total_bytes, 1))
            return vals;

        vals.push_back(
            read_string(data, byte_position, total_bytes, str_len));

        if(!advance<char>(data, byte_position, total_bytes, str_len))
            return vals;
    }
    return vals;
}

/*!
*   \brief Unpack a buffer of scalar values and return
*          a vector of scalars
*   \param buf The data buffer
*   \tparam T the data type associated with the values in the
*           buffer
*   \return std::vector<T> of the buffer values
*/
template <class T>
extern inline std::vector<T> unpack_scalar_buf(
    const std::string_view& buf)
{
    void* data = (void*)(buf.data());

    size_t byte_position = 0;
    size_t total_bytes = buf.size();

    type_t type = read<type_t>(data, byte_position, total_bytes);

    if(!advance<type_t>(data, byte_position,
                         total_bytes, 1))
        return std::vector<T>();

    if( (total_bytes - byte_position) % sizeof(T))
        throw std::runtime_error("The data portion of the provided "\
                                 "metadata buffer does not contain "
                                 "the correct number of bytes for "
                                 "a " + std::to_string(sizeof(T)) +
                                 " byte scalar type. It contains " +
                                 std::to_string(total_bytes -
                                 byte_position) + " bytes");

    size_t n_vals = (total_bytes - byte_position) / sizeof(T);
    std::vector<T> vals(n_vals);

    std::memcpy(vals.data(), (T*)data, total_bytes - byte_position);
    return vals;
}

} //namespace MetadataBuffer

#endif
