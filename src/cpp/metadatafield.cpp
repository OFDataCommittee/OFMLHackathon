#include "metadatafield.h"

using namespace SmartRedis;

MetadataField::MetadataField(const std::string& name,
                             MetaDataType type)
{
    this->_name = name;
    this->_type = type;
}

std::string MetadataField::name()
{
    return this->_name;
}

MetaDataType MetadataField::type()
{
    return this->_type;
}