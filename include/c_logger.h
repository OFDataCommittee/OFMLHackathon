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

#ifndef SMARTREDIS_C_LOGGER_H
#define SMARTREDIS_C_LOGGER_H

#include <stdlib.h>
#include "sr_enums.h"
#include "srobject.h"

///@file

#ifdef __cplusplus
extern "C" {
#endif

/*
* Redirect logging functions to exception-free variants for C and Fortran
*/
#ifndef __cplusplus
#define log_data log_data_noexcept
#define log_warning log_warning_noexcept
#define log_error log_error_noexcept
#define log_data_string log_data_noexcept_string
#define log_warning_string log_warning_noexcept_string
#define log_error_string log_error_noexcept_string
#endif // __cplusplus

/*!
*   \brief Conditionally log data if the logging level is high enough
*   \param context Object containing the log context
*   \param level Minimum logging level for data to be logged
*   \param data Text of data to be logged
*   \param data_len Length in characters of data to be logged
*/
void log_data_noexcept(
    const void* context,
    SRLoggingLevel level,
    const char* data,
    size_t data_len);

/*!
*   \brief Conditionally log a warning if the logging level is high enough
*   \param context Object containing the log context
*   \param level Minimum logging level for data to be logged
*   \param data Text of data to be logged
*   \param data_len Length in characters of data to be logged
*/
void log_warning_noexcept(
    const void* context,
    SRLoggingLevel level,
    const char* data,
    size_t data_len);

/*!
*   \brief Conditionally log an error if the logging level is high enough
*   \param context Object containing the log context
*   \param level Minimum logging level for data to be logged
*   \param data Text of data to be logged
*   \param data_len Length in characters of data to be logged
*/
void log_error_noexcept(
    const void* context,
    SRLoggingLevel level,
    const char* data,
    size_t data_len);

/*!
*   \brief Conditionally log data if the logging level is high enough,
*          using a string to provide a custom context
*   \param context Null-terminated string containing the log context
*   \param context_len Length in characters of the logging context
*   \param level Minimum logging level for data to be logged
*   \param data Text of data to be logged
*   \param data_len Length in characters of data to be logged
*/
void log_data_noexcept_string(
    const char* context,
    size_t context_len,
    SRLoggingLevel level,
    const char* data,
    size_t data_len);

/*!
*   \brief Conditionally log a warning if the logging level is high enough,
*          using a string to provide a custom context
*   \param context Null-terminated string containing the log context
*   \param context_len Length in characters of the logging context
*   \param level Minimum logging level for data to be logged
*   \param data Text of data to be logged
*   \param data_len Length in characters of data to be logged
*/
void log_warning_noexcept_string(
    const char* context,
    size_t context_len,
    SRLoggingLevel level,
    const char* data,
    size_t data_len);

/*!
*   \brief Conditionally log an error if the logging level is high enough,
*          using a string to provide a custom context
*   \param context Null-terminated string containing the log context
*   \param context_len Length in characters of the logging context
*   \param level Minimum logging level for data to be logged
*   \param data Text of data to be logged
*   \param data_len Length in characters of data to be logged
*/
void log_error_noexcept_string(
    const char* context,
    size_t context_len,
    SRLoggingLevel level,
    const char* data,
    size_t data_len);

#ifdef __cplusplus
} // extern "C"
#endif


#endif // SMARTREDIS_C_LOGGER_H
