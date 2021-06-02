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

#include "enums/cpp_metadata_type.h"
#include <string>
#include <vector>
#include <iostream>

namespace SmartRedis {

/*!
*   \brief  The MetadataBuffer class controls the logic
*           for retrieving values out of the buffer.  Note
*           that the MetadataBuffer class does not manage
*           the memory associated with the buffer
*           used for construction, so if that buffer is
*           destroyed, the MeatdataBuffer will not function
*           correctly.  MetadataBuffer also contains
*           static member functions for creating buffers.
*           The serialization and deserialization of
*           buffers assumes 8 bit bytes.
*/
class MetadataBuffer {

    public:

        /*!
        *   \brief MetadataBuffer constructor
        *   \param buf The metadata buffer
        */
        MetadataBuffer(const std::string_view& buf);


        /*!
        *   \brief Default MetadataBuffer copy constructor
        *   \param m_buf The MetadataBuffer to be copied.
        */
        MetadataBuffer(const MetadataBuffer& m_buf) = default;

        /*!
        *   \brief Default MetadataBuffer move constructor
        *   \param m_buf The MetadataBuffer  to be moved for
        *                construction.
        */
        MetadataBuffer(MetadataBuffer&& m_buf) = default;

        /*!
        *   \brief Default MetadataBuffer copy assignment operator
        *   \param m_buf The MetadataBuffer to be copied.
        */
        MetadataBuffer& operator=(const MetadataBuffer& m_buf)
            = default;

        /*!
        *   \brief Default MetadataBuffer move assignment operator
        *   \param m_buf The MetadataBuffer to be moved.
        */
        MetadataBuffer& operator=(MetadataBuffer&& m_buf) = default;

        /*!
        *   \brief Default MetadataBuffer destructor
        */
        virtual ~MetadataBuffer() = default;

        /*!
        *   \brief Default MetadataBuffer move assignment operator
        *   \param buf The metadata buffer
        *   \return The MetaDataType embedded in the buffer
        */
        static MetaDataType get_type(const std::string_view& buf);

        /*!
        *   \brief Generate a metadata buffer for a vector of
        *          scalar values
        *   \param type The MetaDataType associated with the values
        *   \param data The metadata values
        *   \tparam The scalar type
        *   \return A std::string containing the metadata buffer
        */
        template <typename T>
        static std::string generate_scalar_buf(MetaDataType type,
                                               const std::vector<T>& data);

        /*!
        *   \brief Generate a metadata buffer for a vector of
        *          strings
        *   \param data The metadata values
        *   \return A std::string containing the metadata buffer
        */
        static std::string generate_string_buf(
            const std::vector<std::string>& data);

        /*!
        *   \brief Unpack a metadata string buffer and return the
        *          string values
        *   \param buf The metadata buffer
        *   \return A std::vector<std::string> of the buffer values
        */
        static std::vector<std::string> unpack_string_buf(
            const std::string_view& buf);

        /*!
        *   \brief Unpack a metadata scalar buffer and return the
        *          scalar values
        *   \param buf The metadata buffer
        *   \tparam T The scalar type
        *   \return A std::vector<T> of the buffer values
        */
        template <typename T>
        static std::vector<T> unpack_scalar_buf(
            const std::string_view& buf);

        /*!
        *   \brief The MetaDataType associated with the buffer
        *   \return The MetaDataType embedded in the buffer
        */
        static MetaDataType type();

        /*!
        *   \brief The data type for the MetaDataType embedded
        *          in the buffer
        */
        typedef int8_t type_t;

        /*!
        *   \brief The data type for the the length or number
        *          of entries.
        */
        typedef size_t length_t;


    private:

        /*!
        *   \brief The std::string_view containing the buffer
        *          data.  Note that this memory is not
        *          managed by the MetadataBuffer object,
        *          so if it goes out of scope, then the
        *          object will not function correctly.
        */
        std::string_view _buf;

        /*!
        *   \brief A c-ptr to the current location
        *          in the buffer.
        */
        void* _data_pos;


        /*!
        *   \brief The current index in the data buffer
        *          as it is being parsed (in bytes).
        */
        size_t _byte_index;

        /*!
        *   \brief The size of the buffer (in bytes).
        */
        size_t _total_bytes;

        /*!
        *   \brief Put the buffer characters into the
        *          buffer string.
        *   \param buf The buffer in which the characters
        *              should be placed.
        *   \param pos The position in the buffer to place
        *              characters.
        *   \param buf_chars The characters to place in the
        *                    buffer.
        *   \param n_chars The number of characters to place
        *                  in the buffer.
        */
        static void _add_buf_data(std::string& buf,
                                         size_t pos,
                                         const void* data,
                                         size_t n_bytes);

        /*!
        *   \brief Reads a single value of type T from the
        *          the buffer.
        *   \param buf A c-ptr to the buffer
        *   \param byte_position The byte position in the buffer
        *   \param total_bytes The total number of bytes in the
        *                      buffer
        *   \param n_values The number of values to be read.
        *   \throw std::runtime_error if it is not safe to read
        *          (i.e. there is an error in the buffer).
        */
        template <typename T>
        static T _read(void* buf,
                              const size_t& byte_position,
                              const size_t& total_bytes,
                              const size_t& n_values);

        /*!
        *   \brief Checks to see if it is safe
        *          to read the bytes in the buffer
        *          required for n_values of type T.
        *   \param byte_position The byte position in the buffer
        *   \param total_bytes The total number of bytes in the
        *                      buffer
        *   \param n_values The number of values to be read.
        *   \throw std::runtime_error if it is not safe to read
        *          (i.e. there is an error in the buffer).
        */
        template <typename T>
        static bool _safe_to_read(const size_t& byte_position,
                                         const size_t& total_bytes,
                                         const size_t& n_values);

        /*!
        *   \brief Advance the buffer pointer n values of type T.
        *          If the buffer cannot be advanced that many values,
        *          false will be returned and byte_position
        *          will be set to the value of total_bytes.
        *   \param buf A c-ptr to the buffer that is advanced
        *   \param byte_position The byte position in the buffer
        *                        that is updated.
        *   \param total_bytes The total number of bytes in the
        *                      buffer
        *   \param n_values The number of values to advance
        *   \return True if the byte_position can be advanced
        *           n values without reaching the end (or beyond)
        *           of the buffer.  Otherwise, false.
        */
        template <typename T>
        static bool _advance(void*& buf,
                                    size_t& byte_position,
                                    const size_t& total_bytes,
                                    const size_t& n_values);

        /*!
        *   \brief Reads a string from the buffer
        *   \param buf A c-ptr to the buffer
        *   \param byte_position The byte position in the buffer
        *   \param total_bytes The total number of bytes in the
        *                      buffer
        *   \param n_values The number of characters to read
        *   \throw std::runtime_error if it is not safe to read
        *          (i.e. there is an error in the buffer).
        */
        static std::string _read_string(void* buf,
                                               const size_t& byte_position,
                                               const size_t& total_bytes,
                                               const size_t& n_chars);
};

MetadataBuffer::MetadataBuffer(const std::string_view& buf)
{

    this->_total_bytes = sizeof(char)*buf.size();

    if(this->_total_bytes <= sizeof(type_t))
        throw std::runtime_error("The buffer does not contain "\
                                 "any values.  It may be corrupt.");

    this->_buf = buf;
    this->_data_pos = (void*)buf.data();
    this->_byte_index = 0;
}


MetaDataType MetadataBuffer::get_type(const std::string_view& buf)
{
    if(buf.size()*sizeof(char) < sizeof(size_t))
        throw std::runtime_error("The MetadataField type cannot "\
                                 "be retrived from buffer of " +
                                 std::to_string(buf.size()) +
                                 "characters.");

    void* data = (void*)(buf.data());
    int8_t type = *((int8_t*)data);
    return (MetaDataType)type;
}

template <typename T>
std::string MetadataBuffer::generate_scalar_buf(MetaDataType type,
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
    MetadataBuffer::_add_buf_data(buf, pos, &type_id, n_bytes);
    pos += n_bytes;

    // Add the values
    n_bytes = sizeof(T) * data.size();
    const T* v_data = data.data();
    MetadataBuffer::_add_buf_data(buf, pos, v_data, n_bytes);
    return buf;
}

std::string MetadataBuffer::generate_string_buf(
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
    MetadataBuffer::_add_buf_data(buf, pos, &type_id, n_bytes);
    pos += n_bytes;

    // Add each string length and string value
    size_t str_length;
    size_t entry_bytes;
    for(size_t i=0; i<data.size(); i++) {
        str_length = data[i].size();
        entry_bytes = sizeof(size_t);
        MetadataBuffer::_add_buf_data(buf, pos, &str_length, entry_bytes);
        pos += entry_bytes;

        MetadataBuffer::_add_buf_data(buf, pos,
                                      (void*)(data[i].data()),
                                      data[i].size());
        pos += data[i].size();
    }
    return buf;
}

std::vector<std::string> MetadataBuffer::unpack_string_buf(
    const std::string_view& buf)
{
    size_t byte_position = 0;
    size_t total_bytes = buf.size();

    if(total_bytes==0)
        return std::vector<std::string>();

    void* data = (void*)(buf.data());

    type_t type = MetadataBuffer::_read<type_t>(data,
                                                byte_position,
                                                total_bytes, 1);


    if(type!=(type_t)MetaDataType::string)
        throw std::runtime_error("The buffer string metadata type "\
                                 "does not contain the expected "\
                                 "type of string.");

    std::vector<std::string> vals;


    if(!MetadataBuffer::_advance<type_t>(data, byte_position,
                                         total_bytes, 1))
        return vals;

    size_t str_len;

    while(byte_position < total_bytes) {

        str_len = MetadataBuffer::_read<size_t>(data, byte_position,
                                                total_bytes, 1);

        if(!MetadataBuffer::_advance<size_t>(data, byte_position,
                                             total_bytes, 1))
            return vals;

        vals.push_back(
            MetadataBuffer::_read_string(data, byte_position,
                                         total_bytes, str_len));

        if(!MetadataBuffer::_advance<char>(data, byte_position,
                                           total_bytes, str_len))
            return vals;
    }
    return vals;
}


void MetadataBuffer::_add_buf_data(std::string& buf,
                                          size_t pos,
                                          const void* data,
                                          size_t n_bytes)
{
    std::memcpy(buf.data()+pos, data, n_bytes);
    return;
}

template <typename T>
T MetadataBuffer::_read(void* buf,
                               const size_t& byte_position,
                               const size_t& total_bytes,
                               const size_t& n_values)
{
    MetadataBuffer::_safe_to_read<T>(byte_position, total_bytes, n_values);
    return *((T*)buf);
}

template <typename T>
bool MetadataBuffer::_safe_to_read(const size_t& byte_position,
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
bool MetadataBuffer::_advance(void*& buf,
                                     size_t& byte_position,
                                     const size_t& total_bytes,
                                     const size_t& n_values)
{
    // TODO Check overflow error of additions
    size_t final_byte_position = byte_position +
                                 n_values * sizeof(T);
    std::cout<<"Byte position "<<byte_position<<std::endl;
    std::cout<<"Final byte position "<<final_byte_position<<std::endl;
    std::cout<<"Total bytes = "<<total_bytes<<std::endl;
    if(final_byte_position >= total_bytes)
        return false;
    byte_position = final_byte_position;
    buf = (T*)buf + n_values;
    return true;
}

std::string MetadataBuffer::_read_string(void* buf,
                                                const size_t& byte_position,
                                                const size_t& total_bytes,
                                                const size_t& n_chars)
{
    MetadataBuffer::_safe_to_read<char>(byte_position, total_bytes, n_chars);
    return std::string((char*)buf, n_chars);
}

template <class T>
std::vector<T> MetadataBuffer::unpack_scalar_buf(const std::string_view& buf)
{
    void* data = (void*)(buf.data());

    size_t byte_position = 0;
    size_t total_bytes = buf.size();

    type_t type = MetadataBuffer::_read<type_t>(data,
                                                byte_position,
                                                total_bytes, 1);


    if( (total_bytes - byte_position) % sizeof(T))
        throw std::runtime_error("The data portion of the provided "\
                                 "metadata buffer does not contain "
                                 "the correct number of bytes for "
                                 "a " + std::to_string(sizeof(T)) +
                                 " byte scalar type. It contains " +
                                 std::to_string(total_bytes -
                                 byte_position) + " bytes");

    if(!MetadataBuffer::_advance<type_t>(data, byte_position,
                                         total_bytes, 1))
        return;

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

} //namespace SmartRedis

#endif //SMARTREDIS_METADATABUFFER_H