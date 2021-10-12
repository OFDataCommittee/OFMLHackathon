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
    switch (type) {             
        case CTensorType::c_invalid_tensor:
            return TensorType::undefined;
        case CTensorType::c_dbl:
            return TensorType::dbl;
        case CTensorType::c_flt:
            return TensorType::flt;
        case CTensorType::c_int8:
            return TensorType::int8;
        case CTensorType::c_int16:
            return TensorType::int16;
        case CTensorType::c_int32:
            return TensorType::int32;
        case CTensorType::c_int64:
            return TensorType::int64;
        case CTensorType::c_uint8:
            return TensorType::uint8;
        case CTensorType::c_uint16:
            return TensorType::uint16;
        default:
            throw std::runtime_error("Error converting CTensorType "\
                                     "to TensorType.");
    }
}

//! Helper method to convert between CTensorType and TensorType
inline CTensorType convert_tensor_type(TensorType type) {
    switch (type) {
        case TensorType::undefined:
            return CTensorType::c_invalid_tensor;
        case TensorType::dbl:
            return CTensorType::c_dbl;
        case TensorType::flt:
            return CTensorType::c_flt;
        case TensorType::int8:
            return CTensorType::c_int8;
        case TensorType::int16:
            return CTensorType::c_int16;
        case TensorType::int32:
            return CTensorType::c_int32;
        case TensorType::int64:
            return CTensorType::c_int64;
        case TensorType::uint8:
            return CTensorType::c_uint8;
        case TensorType::uint16:
            return CTensorType::c_uint16;
        default :
            throw std::runtime_error("Error converting TensorType "\
                                     "to CTensorType.");
    }
}

} //namespace SmartRedis

#endif //SMARTSIM_TENSORTYPE_H
