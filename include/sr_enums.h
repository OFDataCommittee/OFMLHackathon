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
    SRMemLayoutInvalid           = 0,
    SRMemLayoutNested            = 1,
    SRMemLayoutContiguous        = 2,
    SRMemLayoutFortranNested     = 3,
    SRMemLayoutFortranContiguous = 4
} SRMemoryLayout;

// Metadata types
typedef enum {
    SRMetadataTypeInvalid = 0,
    SRMetadataTypeDouble  = 1,
    SRMetadataTypeFloat   = 2,
    SRMetadataTypeInt32   = 3,
    SRMetadataTypeInt64   = 4,
    SRMetadataTypeUint32  = 5,
    SRMetadataTypeUint64  = 6,
    SRMetadataTypeString  = 7
} SRMetaDataType;

// Tensor types
typedef enum {
    SRTensorTypeInvalid = 0,
    SRTensorTypeDouble  = 1,
    SRTensorTypeFloat   = 2,
    SRTensorTypeInt8    = 3,
    SRTensorTypeInt16   = 4,
    SRTensorTypeInt32   = 5,
    SRTensorTypeInt64   = 6,
    SRTensorTypeUint8   = 7,
    SRTensorTypeUint16  = 8
} SRTensorType;

#endif // SMARTREDIS_ENUMS_H
