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

/* The TensorType enum specifies
the data type of the tensor.  This
enum should be used by users.
*/

#ifndef SMARTSIM_TENSORTYPE_H
#define SMARTSIM_TENSORTYPE_H

#include "enums/c_tensor_type.h"

namespace SmartRedis {

enum class TensorType{
    undefined = 0,
    dbl       = 1,
    flt       = 2,
    int8      = 3,
    int16     = 4,
    int32     = 5,
    int64     = 6,
    uint8     = 7,
    uint16    = 8 };

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

} //namespace SmartRedis

#endif //SMARTSIM_TENSORTYPE_H
