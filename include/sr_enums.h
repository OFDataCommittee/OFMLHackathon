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


#ifndef SMARTREDIS_ENUMS_H
#define SMARTREDIS_ENUMS_H

// Memory layout of tensor data
typedef enum {
    sr_layout_invalid            = 0,
    sr_layout_nested             = 1,
    sr_layout_contiguous         = 2,
    sr_layout_fortran_nested     = 3,
    sr_layout_fortran_contiguous = 4
} SRMemoryLayout;

// Metadata types
typedef enum {
    sr_meta_invalid = 0,
    sr_meta_dbl     = 1,
    sr_meta_flt     = 2,
    sr_meta_int32   = 3,
    sr_meta_int64   = 4,
    sr_meta_uint32  = 5,
    sr_meta_uint64  = 6,
    sr_meta_string  = 7
} SRMetaDataType;

// Tensor types
typedef enum {
    sr_tensor_invalid = 0,
    sr_tensor_dbl     = 1,
    sr_tensor_flt     = 2,
    sr_tensor_int8    = 3,
    sr_tensor_int16   = 4,
    sr_tensor_int32   = 5,
    sr_tensor_int64   = 6,
    sr_tensor_uint8   = 7,
    sr_tensor_uint16  = 8
} SRTensorType;

#endif // SMARTREDIS_ENUMS_H
