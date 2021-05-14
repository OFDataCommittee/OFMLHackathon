#include "stringfield.h"
#include <iostream>
using namespace SmartRedis;

StringField::StringField(const std::string& name) :
    MetadataField(name, MetaDataType::string)
{
    return;
}

StringField::StringField(const std::string& name,
                         const std::string_view& serial_string) :
    MetadataField(name, serial_string)
{
    this->_unpack(serial_string);
    return;
}

std::string StringField::serialize()
{
    /*
    *   The serialized string has the format described
    *   below.
    *
    *   Field           Field size
    *   -------         ----------
    *   type_id         1 byte
    *
    *   n_strings       sizeof(size_t)
    *
    *   str length      sizeof(size_t)
    *   str content     sizeof(char) * str[i].size()
    *   .               .
    *   .               .
    *   str length      sizeof(size_t)
    *   str content     sizeof(char) * str[i].size()
    */

    if(sizeof(int8_t) != sizeof(char))
        throw std::runtime_error("Metadata is not supported on "\
                                 "systems with char length not "\
                                 "equal to one byte.");

    // Number of bytes needed for the type identifier
    size_t type_bytes = sizeof(int8_t);
    // Number of bytes needed for string count
    size_t str_count_bytes = sizeof(size_t);
    // Number of bytes needed for the string lengths
    size_t str_length_bytes = sizeof(size_t) * this->_vals.size();
    // Number of bytes needed for the string values
    size_t data_bytes = 0;
    for(size_t i=0; i<this->_vals.size(); i++)
        data_bytes += this->_vals[i].size() * sizeof(char);

    size_t n_bytes = type_bytes + str_count_bytes +
                     str_length_bytes + data_bytes;
    std::cout<<"n_bytes = "<<n_bytes<<std::endl;
    size_t n_chars = n_bytes / sizeof(char);
    std::cout<<"n_chars = "<<n_chars<<std::endl;

    std::string buf(n_chars, 0);

    size_t pos = 0;

    // Add the type ID
    int8_t type_id = (int8_t)this->_type;
    n_chars = sizeof(int8_t)/sizeof(char);
    this->_place_buf_chars(buf, pos, (char*)(&type_id), n_chars);
    pos += n_chars;

    // Add the number of strings
    size_t n_str = this->_vals.size();
    n_chars = sizeof(size_t)/sizeof(char);
    this->_place_buf_chars(buf, pos, (char*)(&n_str), n_chars);
    pos += n_chars;
    std::cout<<"buf.size() "<<buf.size()<<std::endl;
    // Add each string length and string value
    size_t str_length;
    for(size_t i=0; i<this->_vals.size(); i++) {
        std::cout<<this->_vals[i]<<std::endl;
        std::cout<<this->_vals[i].size()<<std::endl;
        std::cout<<"pos = "<<pos<<std::endl;
        str_length = this->_vals[i].size();
        n_chars = sizeof(size_t) / sizeof(char);
        this->_place_buf_chars(buf, pos, (char*)(&str_length), n_chars);
        pos += n_chars;
        std::cout<<"buf.size() "<<buf.size()<<std::endl;
        this->_place_buf_chars(buf, pos, (char*)(this->_vals[i].data()), this->_vals[i].size());
        pos += this->_vals[i].size();
    }
    std::cout<<"end buf length = "<<buf.size()<<std::endl;
    return buf;
}

void StringField::append(const std::string& value)
{
    this->_vals.push_back(value);
    return;
}

size_t StringField::size()
{
    return this->_vals.size();
}

void StringField::clear()
{
    this->_vals.clear();
    return;
}

std::vector<std::string> StringField::values()
{
    return std::vector<std::string>(this->_vals);
}

void StringField::_unpack(const std::string_view& buf)
{
    void* data = (void*)(buf.data());

    int8_t type = *((int8_t*)data);
    data = ((int8_t*)data) + 1;
    if(type!=(int8_t)this->type())
        throw std::runtime_error("The buffer string metadata type "\
                                 "does not match the object type "\
                                 "being used to interpret it.");

    size_t n_strings = *((size_t*)(data));
    data = ((size_t*)data) + 1;

    this->_vals = std::vector<std::string>(n_strings);
    size_t str_len;
    for(size_t i=0; i<n_strings; i++) {
        str_len = *((size_t*)(data));
        data = ((size_t*)data) + 1;
        this->_vals[i] = std::string((char*)data, str_len);
        data = (char*)data + str_len;
    }

    return;
}

void StringField::_place_buf_chars(std::string& buf,
                                   size_t pos,
                                   char* buf_chars,
                                   size_t n_chars)
{
    for(size_t i=0; i<n_chars; i++) {
        buf[pos] = *buf_chars;
        pos++;
        buf_chars++;
    }
    return;
}
