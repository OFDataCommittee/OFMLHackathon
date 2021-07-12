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

MetaData::MetaData(const MetaData& metadata)
{
    /* Copy constructor for Metadata
    */
    std::unordered_map<std::string, MetadataField*>::const_iterator
        field_it = metadata._field_map.cbegin();
    std::unordered_map<std::string, MetadataField*>::const_iterator
        field_it_end = metadata._field_map.cend();
    while(field_it != field_it_end) {
        this->_create_field(field_it->first,
                            field_it->second->type());
        this->_deep_copy_field(this->_field_map[field_it->first],
                               field_it->second);
        field_it++;
    }

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

MetaData& MetaData::operator=(const MetaData& metadata)
{
    if(this!=&metadata) {

        this->_delete_fields();

        std::unordered_map<std::string, MetadataField*>::const_iterator
            field_it = metadata._field_map.cbegin();
        std::unordered_map<std::string, MetadataField*>::const_iterator
            field_it_end = metadata._field_map.cend();

        while(field_it != field_it_end) {
            this->_create_field(field_it->first,
                                field_it->second->type());
            this->_deep_copy_field(this->_field_map[field_it->first],
                                   field_it->second);
            field_it++;
        }

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
    return *this;
}

MetaData& MetaData::operator=(MetaData&& metadata)
{
    if(this!=&metadata) {
        this->_delete_fields();

        this->_field_map =
            std::move(metadata._field_map);
        this->_char_array_mem_mgr =
            std::move(metadata._char_array_mem_mgr);
        this->_char_mem_mgr =
            std::move(metadata._char_mem_mgr);
        this->_double_mem_mgr =
            std::move(metadata._double_mem_mgr);
        this->_float_mem_mgr =
            std::move(metadata._float_mem_mgr);
        this->_int64_mem_mgr =
            std::move(metadata._int64_mem_mgr);
        this->_uint64_mem_mgr =
            std::move(metadata._uint64_mem_mgr);
        this->_int32_mem_mgr =
            std::move(metadata._int32_mem_mgr);
        this->_uint32_mem_mgr =
            std::move(metadata._uint32_mem_mgr);
        this->_str_len_mem_mgr =
            std::move(metadata._str_len_mem_mgr);
    }
    return *this;
}

MetaData::~MetaData()
{
    this->_delete_fields();
    return;
}

void MetaData::add_scalar(const std::string& field_name,
                          const void* value,
                          MetaDataType type)

{
    if(!(this->has_field(field_name))) {
         this->_create_field(field_name, type);
    }

    MetadataField* mdf = this->_field_map[field_name];

    MetaDataType existing_type = mdf->type();
    if(existing_type!=type)
        throw std::runtime_error("The existing metadata field "\
                                 "has a different type. ");

    switch(type) {
        case MetaDataType::dbl :
            ((ScalarField<double>*)(mdf))->append(value);
            break;
        case MetaDataType::flt :
            ((ScalarField<float>*)(mdf))->append(value);
            break;
        case MetaDataType::int64 :
            ((ScalarField<int64_t>*)(mdf))->append(value);
            break;
        case MetaDataType::uint64 :
            ((ScalarField<uint64_t>*)(mdf))->append(value);
            break;
        case MetaDataType::int32 :
            ((ScalarField<int32_t>*)(mdf))->append(value);
            break;
        case MetaDataType::uint32 :
            ((ScalarField<uint32_t>*)(mdf))->append(value);
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
    if(!(this->has_field(field_name)))
         this->_create_field(field_name, MetaDataType::string);

    ((StringField*)(this->_field_map[field_name]))->append(value);
    return;
}

void MetaData::get_scalar_values(const std::string& name,
                                void*& data,
                                size_t& length,
                                MetaDataType& type)
{
    if(!this->has_field(name))
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
    if(!this->has_field(name))
        throw std::runtime_error("The metadata field "
                                 + name +
                                 " does not exist.");

    MetaDataType type = this->_field_map[name]->type();

    if(type!=MetaDataType::string)
        throw std::runtime_error("The metadata field " +
                                 name +
                                 " is not a string field.");

    return ((StringField*)(this->_field_map[name]))->values();
}

bool MetaData::has_field(const std::string& field_name)
{
    return (this->_field_map.count(field_name)>0);
}

void MetaData::clear_field(const std::string& field_name)
{
    if(this->has_field(field_name)) {
        this->_field_map[field_name]->clear();
        delete this->_field_map[field_name];
        this->_field_map.erase(field_name);
    }
    return;
}

void MetaData::_create_field(const std::string& field_name,
                             const MetaDataType type)
{
    switch(type) {
        case MetaDataType::string :
            this->_create_string_field(field_name);
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

void MetaData::_deep_copy_field(MetadataField* dest_field,
                                MetadataField* src_field)
{
    MetaDataType type = src_field->type();
    switch(type) {
        case MetaDataType::string :
            *((StringField*)dest_field) =
                *((StringField*)src_field);
            break;
        case MetaDataType::dbl :
            *((ScalarField<double>*)dest_field) =
                *((ScalarField<double>*)src_field);
            break;
        case MetaDataType::flt :
            *((ScalarField<float>*)dest_field) =
                *((ScalarField<float>*)src_field);
            break;
        case MetaDataType::int64 :
            *((ScalarField<int64_t>*)dest_field) =
                *((ScalarField<int64_t>*)src_field);
            break;
        case MetaDataType::uint64 :
            *((ScalarField<uint64_t>*)dest_field) =
                *((ScalarField<uint64_t>*)src_field);
            break;
        case MetaDataType::int32 :
            *((ScalarField<int32_t>*)dest_field) =
                *((ScalarField<int32_t>*)src_field);
            break;
        case MetaDataType::uint32 :
            *((ScalarField<uint32_t>*)dest_field) =
                *((ScalarField<uint32_t>*)src_field);
            break;
    }
    return;
}

template <typename T>
void MetaData::_create_scalar_field(const std::string& field_name,
                                    const MetaDataType type)
{
    MetadataField* mdf = new ScalarField<T>(field_name, type);
    this->_field_map[field_name] = mdf;
    return;
}

void MetaData::_create_string_field(const std::string& field_name)
{
    MetadataField* mdf = new StringField(field_name);
    this->_field_map[field_name] = mdf;
    return;
}

template <typename T>
void MetaData::_get_numeric_field_values(const std::string& name,
                                         void*& data,
                                         size_t& n_values,
                                         SharedMemoryList<T>& mem_list)
{
    // Fetch the type of the field
    MetaDataType type = this->_field_map[name]->type();

    switch(type) {
        case MetaDataType::dbl : {
            ScalarField<double>* sdf =
                ((ScalarField<double>*)(this->_field_map[name]));
            n_values = sdf->size();
            data = (void*)mem_list.allocate(n_values);
            std::memcpy(data, sdf->data(), n_values*sizeof(T));
            }
            break;
        case MetaDataType::flt : {
            ScalarField<float>* sdf =
                ((ScalarField<float>*)(this->_field_map[name]));
            n_values = sdf->size();
            data = (void*)mem_list.allocate(n_values);
            std::memcpy(data, sdf->data(), n_values*sizeof(T));
            }
            break;
        case MetaDataType::int64 : {
            ScalarField<int64_t>* sdf =
                ((ScalarField<int64_t>*)(this->_field_map[name]));
            n_values = sdf->size();
            data = (void*)mem_list.allocate(n_values);
            std::memcpy(data, sdf->data(), n_values*sizeof(T));
            }
            break;
        case MetaDataType::uint64 : {
            ScalarField<uint64_t>* sdf =
                ((ScalarField<uint64_t>*)(this->_field_map[name]));
            n_values = sdf->size();
            data = (void*)mem_list.allocate(n_values);
            std::memcpy(data, sdf->data(), n_values*sizeof(T));
            }
            break;
        case MetaDataType::int32 : {
            ScalarField<int32_t>* sdf =
                ((ScalarField<int32_t>*)(this->_field_map[name]));
            n_values = sdf->size();
            data = (void*)mem_list.allocate(n_values);
            std::memcpy(data, sdf->data(), n_values*sizeof(T));
            }
            break;
        case MetaDataType::uint32 : {
            ScalarField<uint32_t>* sdf =
                ((ScalarField<uint32_t>*)(this->_field_map[name]));
            n_values = sdf->size();
            data = (void*)mem_list.allocate(n_values);
            std::memcpy(data, sdf->data(), n_values*sizeof(T));
            }
            break;
        case MetaDataType::string :
            throw std::runtime_error("Invalid MetaDataType used in "\
                                     "MetaData.add_scalar().");
            break;
    }

    return;
}

std::vector<std::pair<std::string, std::string>>
    MetaData::get_metadata_serialization_map()
{
    std::unordered_map<std::string, MetadataField*>::iterator
        mdf_it = this->_field_map.begin();
    std::unordered_map<std::string, MetadataField*>::iterator
        mdf_it_end = this->_field_map.end();

    std::vector<std::pair<std::string, std::string>> fields;

    while(mdf_it!=mdf_it_end) {
        fields.push_back({mdf_it->first,mdf_it->second->serialize()});
        mdf_it++;
    }

    return fields;
}

void MetaData::add_serialized_field(const std::string& name,
                                    char* buf,
                                    size_t buf_size)
{

    std::string_view buf_sv(buf, buf_size);
    MetaDataType type = MetadataBuffer::get_type(buf_sv);

    if(this->has_field(name))
        throw std::runtime_error("Cannot add serialized field if "\
                                 "already exists.");

    switch(type) {
        case MetaDataType::dbl : {
            MetadataField* mdf = new ScalarField<double>(
                name, MetaDataType::dbl,
                MetadataBuffer::unpack_scalar_buf<double>(buf_sv));
            this->_field_map[name] = mdf;
            }
            break;
        case MetaDataType::flt : {
            MetadataField* mdf = new ScalarField<float>(
                name, MetaDataType::flt,
                MetadataBuffer::unpack_scalar_buf<float>(buf_sv));
            this->_field_map[name] = mdf;
            }
            break;
        case MetaDataType::int64 : {
            MetadataField* mdf = new ScalarField<int64_t>(
                name, MetaDataType::int64,
                MetadataBuffer::unpack_scalar_buf<int64_t>(buf_sv));
            this->_field_map[name] = mdf;
            }
            break;
        case MetaDataType::uint64 : {
            MetadataField* mdf = new ScalarField<uint64_t>(
                name, MetaDataType::uint64,
                MetadataBuffer::unpack_scalar_buf<uint64_t>(buf_sv));
            this->_field_map[name] = mdf;
            }
            break;
        case MetaDataType::int32 : {
            MetadataField* mdf = new ScalarField<int32_t>(
                name, MetaDataType::int32,
                MetadataBuffer::unpack_scalar_buf<int32_t>(buf_sv));
            this->_field_map[name] = mdf;
            }
            break;
        case MetaDataType::uint32 : {
            MetadataField* mdf = new ScalarField<uint32_t>(
                name, MetaDataType::uint32,
                MetadataBuffer::unpack_scalar_buf<uint32_t>(buf_sv));
            this->_field_map[name] = mdf;
            }
            break;
        case MetaDataType::string : {
            MetadataField* mdf = new StringField(
                name, MetadataBuffer::unpack_string_buf(buf_sv));
            this->_field_map[name] = mdf;
            }
            break;
    }
    return;
}

void MetaData::_delete_fields()
{
    std::unordered_map<std::string, MetadataField*>::iterator
        it = this->_field_map.begin();
    std::unordered_map<std::string, MetadataField*>::iterator
        it_end = this->_field_map.end();
    while(it!=it_end) {
        delete it->second;
        it++;
    }
    this->_field_map.clear();
}