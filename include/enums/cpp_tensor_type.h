/* The TensorType enum specifies
the data type of the tensor.  This
enum should be used by users.
*/

#ifndef SMARTSIM_TENSORTYPE_H
#define SMARTSIM_TENSORTYPE_H

#include "enums/c_tensor_type.h"

enum class TensorType{
    dbl    = 1,
    flt    = 2,
    int8   = 3,
    int16  = 4,
    int32  = 5,
    int64  = 6,
    uint8  = 7,
    uint16 = 8 };

//! Helper method to convert between CTensorType and TensorType
inline TensorType convert_tensor_type(CTensorType type) {
    TensorType t;
    switch(type) {
        case CTensorType::c_dbl :
            t = TensorType::dbl;
            break;
        case CTensorType::c_flt :
            t = TensorType::flt;
            break;
        case CTensorType::c_int8 :
            t = TensorType::int8;
            break;
        case CTensorType::c_int16 :
            t = TensorType::int16;
            break;
        case CTensorType::c_int32 :
            t = TensorType::int32;
            break;
        case CTensorType::c_int64 :
            t = TensorType::int64;
            break;
        case CTensorType::c_uint8 :
            t = TensorType::uint8;
            break;
        case CTensorType::c_uint16 :
            t = TensorType::uint16;
            break;
        default :
            throw std::runtime_error("Error converting CTensorType "\
                                     "to TensorType.");
    }
    return t;
}

//! Helper method to convert between CTensorType and TensorType
inline CTensorType convert_tensor_type(TensorType type) {
    CTensorType t;
    switch(type) {
        case TensorType::dbl :
            t = CTensorType::c_dbl;
            break;
        case TensorType::flt :
            t = CTensorType::c_flt;
            break;
        case TensorType::int8 :
            t = CTensorType::c_int8;
            break;
        case TensorType::int16 :
            t = CTensorType::c_int16;
            break;
        case TensorType::int32 :
            t = CTensorType::c_int32;
            break;
        case TensorType::int64 :
            t = CTensorType::c_int64;
            break;
        case TensorType::uint8 :
            t = CTensorType::c_uint8;
            break;
        case TensorType::uint16 :
            t = CTensorType::c_uint16;
            break;
        default :
            throw std::runtime_error("Error converting TensorType "\
                                     "to CTensorType.");
    }
    return t;

}

#endif //SMARTSIM_TENSORTYPE_H