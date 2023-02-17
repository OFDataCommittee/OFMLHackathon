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
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cctype>

#include "utility.h"
#include "logger.h"
#include "srexception.h"
#include "srassert.h"
#include "srobject.h"

using namespace SmartRedis;

// Conditionally log data if the logging level is high enough
// (exception-free variant)
extern "C" void log_data_noexcept(
    const void* context,
    SRLoggingLevel level,
    const char* data,
    size_t data_len)
{
    try {
        SR_CHECK_PARAMS(context != NULL && data != NULL);
        const SRObject* temp_context =
            reinterpret_cast<const SRObject*>(context);
        std::string strData(data, data_len);
        temp_context->log_data(level, strData);
    }
    catch (Exception& e) {
        std::cout << "Logging failure: " << e.where()
                  << ": " << e.what() << std::endl;
    }
}

// Conditionally log a warning if the logging level is high enough
// (exception-free variant)
extern "C" void log_warning_noexcept(
    const void* context,
    SRLoggingLevel level,
    const char* data,
    size_t data_len)
{
    try {
        SR_CHECK_PARAMS(context != NULL && data != NULL);
        const SRObject* temp_context =
            reinterpret_cast<const SRObject*>(context);
        std::string strData(data, data_len);
        temp_context->log_warning(level, strData);
    }
    catch (Exception& e) {
        std::cout << "Logging failure: " << e.where()
                  << ": " << e.what() << std::endl;
    }
}

// Conditionally log an error if the logging level is high enough
// (exception-free variant)
extern "C" void log_error_noexcept(
    const void* context,
    SRLoggingLevel level,
    const char* data,
    size_t data_len)
{
    try {
        SR_CHECK_PARAMS(context != NULL && data != NULL);
        const SRObject* temp_context =
            reinterpret_cast<const SRObject*>(context);
        std::string strData(data, data_len);
        temp_context->log_error(level, strData);
    }
    catch (Exception& e) {
        std::cout << "Logging failure: " << e.where()
                  << ": " << e.what() << std::endl;
    }
}


// Conditionally log data if the logging level is high enough
// (exception-free variant)
extern "C" void log_data_noexcept_string(
    const char* context,
    size_t context_len,
    SRLoggingLevel level,
    const char* data,
    size_t data_len)
{
    try {
        SR_CHECK_PARAMS(context != NULL && data != NULL);
        std::string temp_context(context, context_len);
        std::string strData(data, data_len);
        log_data(temp_context, level, strData);
    }
    catch (Exception& e) {
        std::cout << "Logging failure: " << e.where()
                  << ": " << e.what() << std::endl;
    }
}

// Conditionally log a warning if the logging level is high enough
// (exception-free variant)
extern "C" void log_warning_noexcept_string(
    const char* context,
    size_t context_len,
    SRLoggingLevel level,
    const char* data,
    size_t data_len)
{
    try {
        SR_CHECK_PARAMS(context != NULL && data != NULL);
        std::string temp_context(context, context_len);
        std::string strData(data, data_len);
        log_warning(temp_context, level, strData);
    }
    catch (Exception& e) {
        std::cout << "Logging failure: " << e.where()
                  << ": " << e.what() << std::endl;
    }
}

// Conditionally log an error if the logging level is high enough
// (exception-free variant)
extern "C" void log_error_noexcept_string(
    const char* context,
    size_t context_len,
    SRLoggingLevel level,
    const char* data,
    size_t data_len)
{
    try {
        SR_CHECK_PARAMS(context != NULL && data != NULL);
        std::string temp_context(context, context_len);
        std::string strData(data, data_len);
        log_error(temp_context, level, strData);
    }
    catch (Exception& e) {
        std::cout << "Logging failure: " << e.where()
                  << ": " << e.what() << std::endl;
    }
}
