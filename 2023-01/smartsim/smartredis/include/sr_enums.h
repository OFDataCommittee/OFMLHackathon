/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2023, Hewlett Packard Enterprise
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


#ifndef SMARTREDIS_SR_ENUMS_H
#define SMARTREDIS_SR_ENUMS_H

///@file

/*!
*   \brief  Enumeration for memory layout of tensor data
*/
typedef enum {
    SRMemLayoutInvalid           = 0, // Invalid or uninitialized memory layout
    SRMemLayoutNested            = 1, // Multidimensional row-major array layout with nested arrays of pointers (contiguous at innermost layer)
    SRMemLayoutContiguous        = 2, // Multidimensional row-major array layout in contiguous memory
    SRMemLayoutFortranNested     = 3, // Multidimensional column-major array layout with nested arrays of pointers (contiguous at innermost layer)
    SRMemLayoutFortranContiguous = 4  // Multidimensional column-major array layout in contiguous memory
} SRMemoryLayout;

/*!
*   \brief  Enumeration for metadata types
*/
typedef enum {
    SRMetadataTypeInvalid = 0, // Invalid or uninitialized metadata
    SRMetadataTypeDouble  = 1, // Double-precision floating point metadata
    SRMetadataTypeFloat   = 2, // Floating point metadata
    SRMetadataTypeInt32   = 3, // 32-bit signed integer metadata
    SRMetadataTypeInt64   = 4, // 64-bit signed integer metadata
    SRMetadataTypeUint32  = 5, // 32-bit unsigned integer metadata
    SRMetadataTypeUint64  = 6, // 64-bit unsigned integer metadata
    SRMetadataTypeString  = 7  // ASCII text string metadata
} SRMetaDataType;

/*!
*   \brief  Enumeration for tensor data types
*/
typedef enum {
    SRTensorTypeInvalid = 0, // Invalid or uninitialized tensor type
    SRTensorTypeDouble  = 1, // Double-precision floating point tensor type
    SRTensorTypeFloat   = 2, // Floating point tensor type
    SRTensorTypeInt8    = 3, // 8-bit signed integer tensor type
    SRTensorTypeInt16   = 4, // 16-bit signed integer tensor type
    SRTensorTypeInt32   = 5, // 32-bit signed integer tensor type
    SRTensorTypeInt64   = 6, // 64-bit signed integer tensor type
    SRTensorTypeUint8   = 7, // 8-bit unsigned integer tensor type
    SRTensorTypeUint16  = 8  // 16-bit unsigned integer tensor type
} SRTensorType;

/*!
*   \brief  Enumeration for logging levels
*/
typedef enum {
    LLInvalid   = 0, // Invalid or uninitialized logging level
    LLQuiet     = 1, // No logging at all
    LLInfo      = 2, // Informational logging only
    LLDebug     = 3, // Verbose logging for debugging purposes
    LLDeveloper = 4  // Extra verbose logging for internal use
} SRLoggingLevel;

#endif // SMARTREDIS_SR_ENUMS_H
