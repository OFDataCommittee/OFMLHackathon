#include "metadatafield.h"

using namespace SmartRedis;

MetaDataType MetadataField::get_type_from_buffer(
    const std::string_view& buf)
{
    void* data = (void*)(buf.data());
    int8_t type = *((int8_t*)data);
    return (MetaDataType)type;
}
