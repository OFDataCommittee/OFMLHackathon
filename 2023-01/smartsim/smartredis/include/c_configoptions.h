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

#ifndef SMARTREDIS_C_CONFIGOPTIONS_H
#define SMARTREDIS_C_CONFIGOPTIONS_H

#include "sr_enums.h"
#include "srexception.h"

///@file
///\brief C-wrappers for the C++ DataSet class

#ifdef __cplusplus
extern "C" {
#endif

/////////////////////////////////////////////////////////////
// Factory construction methods

/*!
*   \brief Instantiate ConfigOptions, getting selections from
*          environment variables. If \p db_suffix is non-empty,
*          then "_{db_suffix}" will be appended to the name of
*          each environment variable that is read.
*   \param db_suffix The suffix to use with environment variables,
*                    or an empty string to disable suffixing
*   \param db_suffix_length The length of the db_suffix string,
*                           excluding null terminating character
*   \param new_configoptions Receives the new configoptions object
*   \returns Returns SRNoError on success or an error code on failure
*/
SRError create_configoptions_from_environment(
    const char* db_suffix,
    const size_t db_suffix_length,
    void** new_configoptions);

/////////////////////////////////////////////////////////////
// Option access

/*!
*   \brief Retrieve the value of a numeric configuration option
*          from the selected source
*   \param c_cfgopts The ConfigOptions object to use for communication
*   \param option_name The name of the configuration option to retrieve
*   \param option_name_len The length of the option_name string,
*                          excluding null terminating character
*   \param result Receives the selected integer option result. Returns
*                 \p default_value if the option was not set in the
*                 selected source
*   \returns Returns SRNoError on success, SRKeyError if the configuration
*            option is not set, or an error code on failure
*/
SRError get_integer_option(
    void* c_cfgopts,
    const char* option_name,
    size_t option_name_len,
    int64_t* result);

/*!
*   \brief Retrieve the value of a string configuration option
*          from the selected source
*   \param c_cfgopts The ConfigOptions object to use for communication
*   \param option_name The name of the configuration option to retrieve
*   \param option_name_len The length of the option_name string,
*                          excluding null terminating character
*   \param result Receives the selected integer option result. Returns
*                 \p default_value if the option was not set in the
*                 selected source
*   \param result_len Receives the length of the result string,
*                     excluding null terminating character
*   \returns Returns SRNoError on success, SRKeyError if the configuration
*            option is not set, or an error code on failure
*/
SRError get_string_option(
    void* c_cfgopts,
    const char* option_name,
    size_t option_name_len,
    char** result,
    size_t* result_len);

/*!
*   \brief Check whether a configuration option is set in the
*          selected source
*   \param c_cfgopts The ConfigOptions object to use for communication
*   \param option_name The name of the configuration option to check
*   \param option_name_len The length of the option_name string,
*                          excluding null terminating character
*   \param cfg_result Receives true IFF the option was defined or has been
*                     overridden; false otherwise
*   \returns Returns SRNoError on success or an error code on failure
*/
SRError is_configured(
    void* c_cfgopts,
    const char* key,
    size_t key_len,
    bool* cfg_result);

/////////////////////////////////////////////////////////////
// Option overrides

/*!
*   \brief Override the value of a numeric configuration option
*          in the selected source
*   \details Overrides are specific to an instance of the
*            ConfigOptions class. An instance that references
*            the same source will not be affected by an override to
*            a different ConfigOptions instance
*   \param c_cfgopts The ConfigOptions object to use for communication
*   \param option_name The name of the configuration option to override
*   \param option_name_len The length of the option_name string,
*                          excluding null terminating character
*   \param value The value to store for the configuration option
*   \returns Returns SRNoError on success or an error code on failure
*/
SRError override_integer_option(
    void* c_cfgopts,
    const char* option_name,
    size_t option_name_len,
    int64_t value);

/*!
*   \brief Override the value of a string configuration option
*          in the selected source
*   \details Overrides are specific to an instance of the
*            ConfigOptions class. An instance that references
*            the same source will not be affected by an override to
*            a different ConfigOptions instance
*   \param c_cfgopts The ConfigOptions object to use for communication
*   \param option_name The name of the configuration option to override
*   \param option_name_len The length of the option_name string,
*                          excluding null terminating character
*   \param value The value to store for the configuration option
*   \param value_len The length of the value string,
*                    excluding null terminating character
*   \returns Returns SRNoError on success or an error code on failure
*/
SRError override_string_option(
    void* c_cfgopts,
    const char* option_name,
    size_t option_name_len,
    const char* value,
    size_t value_len);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SMARTREDIS_C_CONFIGOPTIONS_H
