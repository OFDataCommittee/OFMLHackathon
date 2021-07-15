#include "stringfield.h"

using namespace SmartRedis;

StringField::StringField(const std::string& name) :
    MetadataField(name, MetaDataType::string)
{
    return;
}

StringField::StringField(const std::string& name, const std::vector<std::string>& vals) :
    MetadataField(name, MetaDataType::string)
{
    this->_vals = vals;
    return;
}

StringField::StringField(const std::string& name,
                         std::vector<std::string>&& vals) :
    MetadataField(name, MetaDataType::string)
{
    this->_vals = std::move(vals);
    return;
}

std::string StringField::serialize()
{
    return MetadataBuffer::generate_string_buf(this->_vals);
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

const std::vector<std::string>& StringField::immutable_values()
{
    return this->_vals;
}