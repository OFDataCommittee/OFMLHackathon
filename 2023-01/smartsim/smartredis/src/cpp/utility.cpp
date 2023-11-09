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

#include <cstdlib>
#include <cctype>
#include <algorithm>
#include <string>
#include <cstring>
#include <stdio.h>
#include <stdexcept>
#include "srexception.h"
#include "utility.h"
#include "logger.h"

namespace SmartRedis {

// Initialize an integer from an environment variable
void get_config_integer(int& value,
                        const std::string& cfg_key,
                        const int default_value,
                        int flags /* = 0 */)
{
    bool suppress_warning = 0 != (flags & flag_suppress_warning);
    bool keyerror_on_absent = 0 != (flags & throw_on_absent);

    int result = default_value;
    std::string message = "Getting value for " + cfg_key;
    log_data("SmartRedis Library", LLDebug, message);

    char* cfg_val = std::getenv(cfg_key.c_str());
    message = "Retrieved value \"";
    message += cfg_val == NULL ? "<NULL>" : cfg_val;
    message += "\"";
    if ((NULL == cfg_val) && !keyerror_on_absent)
        message += ". Using default value of " + std::to_string(default_value);
    log_data("SmartRedis Library", LLDebug, message);

    if ((cfg_val == NULL) && keyerror_on_absent) {
        std::string msg("No value found for key ");
        msg += cfg_key;
        throw SRKeyException(msg);
    }

    if (cfg_val != NULL && std::strlen(cfg_val) > 0) {
        // Enforce that all characters are digits because std::stoi
        // will truncate a string like "10xy" to 10.
        // We want to guard users from input errors they might have.
        for (char* c = cfg_val; *c != '\0'; c++) {
            if (!isdigit(*c) && !(*c == '-' && c == cfg_val)) {
                throw SRParameterException("The value of " + cfg_key +
                                           " must be a valid number.");
            }
        }

        try {
            result = std::stoi(cfg_val);
        }
        catch (std::invalid_argument& e) {
            throw SRParameterException("The value of " + cfg_key + " could "\
                                       "not be converted to type integer.");
        }
        catch (std::out_of_range& e) {
            throw SRParameterException("The value of " + cfg_key + " is too "\
                                       "large to be stored as an integer "\
                                       "value.");
        }
        catch (std::exception& e) {
            throw SRInternalException("An unexpected error occurred  "\
                                      "while attempting to convert the "\
                                      "environment variable " + cfg_key +
                                      " to an integer.");
        }
    }
    else if (!suppress_warning) {
        log_warning(
            "SmartRedis Library",
            LLDebug,
            "Configuration variable " + cfg_key + " not set"
        );
    }

    value = result;
    message = "Exiting with value \"" + std::to_string(value) + "\"";
    log_data("SmartRedis Library", LLDebug, message);
}

// Initialize a string from an environment variable
void get_config_string(std::string& value,
                       const std::string& cfg_key,
                       const std::string& default_value,
                       int flags /* = 0 */)
{
    bool suppress_warning = 0 != (flags & flag_suppress_warning);
    bool keyerror_on_absent = 0 != (flags & throw_on_absent);

    std::string result = default_value;
    std::string message = "Getting value for " + cfg_key;
    log_data("SmartRedis Library", LLDebug, message);

    char* cfg_val = std::getenv(cfg_key.c_str());
    message = "Retrieved value \"";
    message += cfg_val == NULL ? "<NULL>" : cfg_val;
    message += "\"";
    if ((NULL == cfg_val) && !keyerror_on_absent)
        message += ". Using default value of " + default_value;
    log_data("SmartRedis Library", LLDebug, message);

    if ((cfg_val == NULL) && keyerror_on_absent) {
        std::string msg("No value found for key ");
        msg += cfg_key;
        throw SRKeyException(msg);
    }

    if (cfg_val != NULL && std::strlen(cfg_val) > 0)
        result = cfg_val;
    else if (!suppress_warning) {
        log_warning(
            "SmartRedis Library",
            LLDebug,
            "Configuration variable " + cfg_key + " not set"
        );
    }

    value = result;
    message = "Exiting with value \"" + value + "\"";
    log_data("SmartRedis Library", LLDebug, message);
}

// Create a string representation of a tensor type
std::string to_string(SRTensorType ttype)
{
    switch (ttype) {
        case SRTensorTypeDouble:
            return "double";
        case SRTensorTypeFloat:
            return "float";
        case SRTensorTypeInt8:
            return "8 bit signed integer";
        case SRTensorTypeInt16:
            return "16 bit signed integer";
        case SRTensorTypeInt32:
            return "32 bit signed integer";
        case SRTensorTypeInt64:
            return "64 bit signed integer";
        case SRTensorTypeUint8:
            return "8 bit unsigned integer";
        case SRTensorTypeUint16:
            return "16 bit unsigned integer";
        case SRTensorTypeInvalid:
            // Fall through
        default:
            return "Invalid tensor type";
    }
}

// Create a string representation of a metadata field type
std::string to_string(SRMetaDataType mdtype)
{
    switch (mdtype) {
        case SRMetadataTypeDouble:
            return "double";
        case SRMetadataTypeFloat:
            return "float";
        case SRMetadataTypeInt32:
            return "32 bit signed integer";
        case SRMetadataTypeInt64:
            return "64 bit signed integer";
        case SRMetadataTypeUint32:
            return "32 bit unsigned integer";
        case SRMetadataTypeUint64:
            return "64 bit unsigned integer";
        case SRMetadataTypeString:
            return "string";
        case SRMetadataTypeInvalid:
            // Fall through
        default:
            return "Invalid metadata type";
    }
}

} // namespace SmartRedis
