#ifndef SMARTREDIS_CMETADATATYPE_H
#define SMARTREDIS_CMETADATATYPE_H

/* Defines the metadata types
for c-clients to use as a
metadata type specifier
*/

typedef enum{
    c_meta_dbl    = 1,
    c_meta_flt    = 2,
    c_meta_int32  = 3,
    c_meta_int64  = 4,
    c_meta_uint32 = 5,
    c_meta_uint64 = 6,
    c_meta_string = 7
}CMetaDataType;

#endif //SMARTREDIS_CMETADATATYPE_H