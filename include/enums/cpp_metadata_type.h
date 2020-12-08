#ifndef SMARTSIM_METADATATYPE_H
#define SMARTSIM_METADATATYPE_H

#include "enums/c_metadata_type.h"

enum class MetaDataType {
    dbl     = 1,
    flt     = 2,
    int32   = 3,
    int64   = 4,
    uint32  = 5,
    uint64  = 6,
    string  = 7
};

//! Helper method to convert between MetaDataType and CMetaDataType
inline CMetaDataType convert_metadata_type(MetaDataType type) {
    CMetaDataType t;
    switch(type) {
        case MetaDataType::dbl :
            t = CMetaDataType::c_meta_dbl;
            break;
        case MetaDataType::flt :
            t = CMetaDataType::c_meta_flt;
            break;
        case MetaDataType::int32 :
            t = CMetaDataType::c_meta_int32;
            break;
        case MetaDataType::int64 :
            t = CMetaDataType::c_meta_int64;
            break;
        case MetaDataType::uint32 :
            t = CMetaDataType::c_meta_uint32;
            break;
        case MetaDataType::uint64 :
            t = CMetaDataType::c_meta_uint64;
            break;
        case MetaDataType::string :
            t = CMetaDataType::c_meta_string;
            break;
        default :
            throw std::runtime_error("Error converting MetaDataType "\
                                     "to CMetaDataType.");
    }
    return t;
}

//! Helper method to convert between CMetaDataType and MetaDataType
inline MetaDataType convert_metadata_type(CMetaDataType type) {
    MetaDataType t;
    switch(type) {
        case CMetaDataType::c_meta_dbl :
            t = MetaDataType::dbl;
            break;
        case CMetaDataType::c_meta_flt :
            t = MetaDataType::flt;
            break;
        case CMetaDataType::c_meta_int32 :
            t = MetaDataType::int32;
            break;
        case CMetaDataType::c_meta_int64 :
            t = MetaDataType::int64;
            break;
        case CMetaDataType::c_meta_uint32 :
            t = MetaDataType::uint32;
            break;
        case CMetaDataType::c_meta_uint64 :
            t = MetaDataType::uint64;
            break;
        case CMetaDataType::c_meta_string :
            t = MetaDataType::string;
            break;
        default :
            throw std::runtime_error("Error converting CMetaDataType "\
                                     "to MetaDataType.");
    }
    return t;
}


#endif //SMARTSIM_METADATATYPE_H