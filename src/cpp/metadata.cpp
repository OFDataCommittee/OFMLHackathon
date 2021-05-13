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

#include "metadata.h"

using namespace SmartRedis;

MetaData::MetaData(const MetaData& metadata) {
    /* Copy constructor for Metadata
    */

    //TODO we need to do a deep copy of the _field_map
    //because those MetadataField values are allocated
    //on the heap
    this->_char_array_mem_mgr = metadata._char_array_mem_mgr;
    this->_char_mem_mgr = metadata._char_mem_mgr;
    this->_double_mem_mgr = metadata._double_mem_mgr;
    this->_float_mem_mgr = metadata._float_mem_mgr;
    this->_int64_mem_mgr = metadata._int64_mem_mgr;
    this->_uint64_mem_mgr = metadata._uint64_mem_mgr;
    this->_int32_mem_mgr = metadata._int32_mem_mgr;
    this->_uint32_mem_mgr = metadata._uint32_mem_mgr;
    this->_str_len_mem_mgr = metadata._str_len_mem_mgr;
}

MetaData& MetaData::operator=(const MetaData& metadata) {

    //TODO we need to do a deep copy of the _field_map
    if(this!=&metadata) {
        this->_pb_metadata_msg = metadata._pb_metadata_msg;
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

void MetaData::add_scalar(const std::string& field_name,
                          const void* value,
                          MetaDataType type)

{
    if(!(this->_field_exists(field_name))) {
         this->_create_field(field_name, type);
    }

    MetadataField* mdf = this->_field_map[field_name];

    MetaDataType existing_type = mdf->type();
    if(existing_type!=type)
        throw std::runtime_error("The existing metadata field "\
                                 "has a different type. ");

    switch(type) {
        case MetaDataType::dbl :
            (ScalarField<double>*)(mdf)->append(value);
            break;
        case MetaDataType::flt :
            (ScalarField<float>*)(mdf)->append(value);
            break;
        case MetaDataType::int64 :
            (ScalarField<int64_t>*)(mdf)->append(value);
            break;
        case MetaDataType::uint64 :
            (ScalarField<uint64_t>*)(mdf)->append(value);
            break;
        case MetaDataType::int32 :
            (ScalarField<int32_t>*)(mdf)->append(value);
            break;
        case MetaDataType::uint32 :
            (ScalarField<uint32_t>*)(mdf)->append(value);
            break;
        case MetaDataType::string :
            throw std::runtime_error("Invalid MetaDataType used in "\
                                     "MetaData.add_scalar().");
            break;
    }
    return;
}

void MetaData::add_string(const std::string& field_name,
                          const std::string& value)
{
    if(!(this->_field_exists(field_name)))
         this->_create_message(field_name, MetaDataType::string);

    ((StringField*)(this->_field_map[field_name]))->append(value);
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

    type = this->_field_map[name]->type();

    switch(type) {
        case MetaDataType::dbl :
            this->_get_numeric_field_values<double>
                (name, data, length, this->_double_mem_mgr);
            break;
        case MetaDataType::flt :
            this->_get_numeric_field_values<float>
                (name, data, length, this->_float_mem_mgr);
            break;
        case MetaDataType::int64 :
            this->_get_numeric_field_values<int64_t>
                (name, data, length, this->_int64_mem_mgr);
            break;
        case MetaDataType::uint64 :
            this->_get_numeric_field_values<uint64_t>
                (name, data, length, this->_uint64_mem_mgr);
            break;
        case MetaDataType::int32 :
            this->_get_numeric_field_values<int32_t>
                (name, data, length, this->_int32_mem_mgr);
            break;
        case MetaDataType::uint32 :
            this->_get_numeric_field_values<uint32_t>
                (name, data, length, this->_uint32_mem_mgr);
            break;
        case MetaDataType::string :
            throw std::runtime_error("MetaData.get_scalar_values() "\
                                     "requested invalid MetaDataType.");
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

    MetaDataType type = this->_field_map[name]->type();

    if(type!=MetaDataType::string)
        throw std::runtime_error("The metadata field " +
                                 name +
                                 " is not a string field.");

    return (StringField*)(this->_field_map[name])->values();
}

void MetaData::clear_field(const std::string& field_name)
{
    if(this->_field_exists(field_name))
        this->_field_map[field_name]->clear();
    return;
}

void MetaData::_create_field(const std::string& field_name,
                             const MetaDataType type)
{
    switch(type) {
        case MetaDataType::string :
            /*
            TODO implement this function
            this->_create_string_field
                    <std::string>(field_name,type);
            */
            break;
        case MetaDataType::dbl :
            this->_create_scalar_field<double>(field_name,type);
            break;
        case MetaDataType::flt :
            this->_create_scalar_field<float>(field_name,type);
            break;
        case MetaDataType::int64 :
            this->_create_scalar_field<int64_t>(field_name,type);
            break;
        case MetaDataType::uint64 :
            this->_create_scalar_field<uint64_t>(field_name,type);
            break;
        case MetaDataType::int32 :
            this->_create_scalar_field<int32_t>(field_name,type);
            break;
        case MetaDataType::uint32 :
            this->_create_scalar_field<uint32_t>(field_name,type);
            break;
    }
    return;
}

template <typename T>
void MetaData::_create_scalar_field<T>(const std::string& field_name,
                                       const MetaDataType type)
{
    MetadataField* mdf = new ScalarField<T>(field_name, type);
    this->_field_map[field_name] = mdf;
    return;
}

void MetaData::_create_string_field(const std::string& field_name)
{
    MetadataField* mdf = new StringField(field_name, type);
    this->_field_map[field_name] = mdf;
    return;
}

template <typename T>
void MetaData::_get_numeric_field_values(const std::string& name,
                                         void*& data,
                                         size_t& n_values,
                                         SharedMemoryList<T>& mem_list)
{
    // Fetch the ScalarField from the map
    ScalarField* sdf = (ScalarField*)(this->_field_map[name]);

    // Get the number of entries and number of bytes
    n_values = sdf->size();

    // Allocate the memory
    data = (void*)mem_list.allocate(n_values);

    // Copy over the field values
    std::memcpy(data, sdf->data(), n_values*sizeof(T));

    return;
}

bool MetaData::_field_exists(const std::string& field_name)
{
   return (this->_field_map.count(field_name)>0);
}