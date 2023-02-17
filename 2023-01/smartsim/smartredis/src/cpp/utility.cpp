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
#include <string>
#include <cstring>
#include <stdio.h>
#include <stdexcept>
#include "srexception.h"
#include "utility.h"
#include "logger.h"

namespace SmartRedis {

/*!
*   \brief Initialize an integer from configuration, such as an
*          environment variable
*   \param value Receives the configuration value
*   \param cfg_key The key to query for the configuration variable
*   \param default_value Default if configuration key is not set
*   \param suppress_warning Do not issue a warning if the variable
*                           is not set
*/
void get_config_integer(int& value,
                        const std::string& cfg_key,
                        const int default_value,
                        bool suppress_warning /*= false*/)
{
    value = default_value;
    std::string message = "Getting value for " + cfg_key;
    log_data("SmartRedis Library", LLDebug, message);

    char* cfg_val = std::getenv(cfg_key.c_str());
    message = "Retrieved value \"";
    message += cfg_val == NULL ? "<NULL>" : cfg_val;
    message += "\"";
    if (NULL == cfg_val)
        message += ". Using default value of " + std::to_string(default_value);
    log_data("SmartRedis Library", LLDebug, message);

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
            value = std::stoi(cfg_val);
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

    message = "Exiting with value \"" + std::to_string(value) + "\"";
    log_data("SmartRedis Library", LLDebug, message);
}

/*!
*   \brief Initialize an string from configuration, such as an
*          environment variable
*   \param value Receives the configuration value
*   \param cfg_key The key to query for the configuration variable
*   \param default_value Default if configuration key is not set
*   \param suppress_warning Do not issue a warning if the variable
*                           is not set
*/
void get_config_string(std::string& value,
                       const std::string& cfg_key,
                       const std::string& default_value,
                       bool suppress_warning /*= false*/)
{
    value = default_value;
    std::string message = "Getting value for " + cfg_key;
    log_data("SmartRedis Library", LLDebug, message);

    char* cfg_val = std::getenv(cfg_key.c_str());
    message = "Retrieved value \"";
    message += cfg_val == NULL ? "<NULL>" : cfg_val;
    message += "\"";
    if (NULL == cfg_val)
        message += ". Using default value of \"" + default_value + "\"";
    log_data("SmartRedis Library", LLDebug, message);

    if (cfg_val != NULL && std::strlen(cfg_val) > 0)
        value = cfg_val;
    else if (!suppress_warning) {
        log_warning(
            "SmartRedis Library",
            LLDebug,
            "Configuration variable " + cfg_key + " not set"
        );
    }

    message = "Exiting with value \"" + value + "\"";
    log_data("SmartRedis Library", LLDebug, message);
}

} // namespace SmartRedis {
