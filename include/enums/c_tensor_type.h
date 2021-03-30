/* Defines the tensor types
for c-clients to use as a
type specifier
*/

#ifndef SMARTREDIS_CTENSORTYPE_H
#define SMARTREDIS_CTENSORTYPE_H

typedef enum{
    c_dbl    = 1,
    c_flt    = 2,
    c_int8   = 3,
    c_int16  = 4,
    c_int32  = 5,
    c_int64  = 6,
    c_uint8  = 7,
    c_uint16 = 8
}CTensorType;

#endif //SMARTREDIS_CTENSORTYPE_H