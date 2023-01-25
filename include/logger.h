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

#ifndef SMARTREDIS_LOGGER_H
#define SMARTREDIS_LOGGER_H

#ifdef __cplusplus // Skip C++ headers for C users
#include <string>
#include "utility.h"
#endif // __cplusplus
#include <stdlib.h>
#include "sr_enums.h"
#include "srobject.h"

///@file

#ifdef __cplusplus
namespace SmartRedis {

class SRObject;

/*!
*   \brief The Logger class implements a logging facility for SmartRedis
*/
class Logger {

    private:

        /*!
        *   \brief Logger constructor. Do not use! To obtain the reference
        *          to the singleton Logger, use the get_instance() method
        */
        Logger() { _initialized = false; }

        /*!
        *   \brief Logger copy constructor unavailable
        *   \param logger The Logger to copy for construction
        */
        Logger(const Logger& logger) = delete;

        /*!
        *   \brief Logger assignment operator unavailable
        *   \param logger The Logger to assign
        */
        void operator=(const Logger& logger) = delete;

        /*!
        *   \brief Logger move assignment operator unavailable
        *   \param logger The Logger to move
        */
        void operator=(Logger&& logger) = delete;

        /*!
        *   \brief Default Logger destructor
        */
        ~Logger() = default;

    public:

        /*!
        *   \brief Retrieve the unique Logger instance
        *   \returns The actual logger instance
        */
       static Logger& get_instance()
       {
           static Logger __instance; // instantiated on first acccess
           return __instance;
       }

        /*!
        *   \brief Set up logging for the current client
        */
        void configure_logging();

        /*!
        *   \brief Conditionally log data if the logging level is high enough
        *   \param context Logging context (string to prefix the log entry with)
        *   \param level Minimum logging level for data to be logged
        *   \param data Text of data to be logged
        */
        void log_data(
            const std::string& context,
            SRLoggingLevel level,
            const std::string& data);

        /*!
        *   \brief Conditionally log warning data if the logging level is
        *          high enough
        *   \param context Logging context (string to prefix the log entry with)
        *   \param level Minimum logging level for data to be logged
        *   \param data Text of data to be logged
        */
        void log_warning(
            const std::string& context,
            SRLoggingLevel level,
            const std::string& data)
        {
            log_data(context, level, "WARNING: " + data);
        }

        /*!
        *   \brief Conditionally log error data if the logging level is
        *          high enough
        *   \param context Logging context (string to prefix the log entry with)
        *   \param level Minimum logging level for data to be logged
        *   \param data Text of data to be logged
        */
        void log_error(
            const std::string& context,
            SRLoggingLevel level,
            const std::string& data)
        {
            log_data(context, level, "ERROR: " + data);
        }

    private:

        /*!
        *   \brief Track whether logging is initialized
        */
        bool _initialized;

        /*!
        *   \brief The file to which to write log data
        */
        std::string _logfile;

        /*!
        *   \brief The current logging level
        */
        SRLoggingLevel _log_level;
};

/*!
*   \brief Conditionally log data if the logging level is high enough
*   \param context Logging context (string to prefix the log entry with)
*   \param level Minimum logging level for data to be logged
*   \param data Text of data to be logged
*/
inline void log_data(
    const std::string& context,
    SRLoggingLevel level,
    const std::string& data)
{
    Logger::get_instance().log_data(context, level, data);
}

/*!
*   \brief Conditionally log data if the logging level is high enough
*   \param context Object containing the log context
*   \param level Minimum logging level for data to be logged
*   \param data Text of data to be logged
*/
inline void log_data(
    const SRObject* context,
    SRLoggingLevel level,
    const std::string& data)
{
    context->log_data(level, data);
}

/*!
*   \brief Conditionally log a warning if the logging level is high enough
*   \param context Logging context (string to prefix the log entry with)
*   \param level Minimum logging level for data to be logged
*   \param data Text of data to be logged
*/
inline void log_warning(
    const std::string& context,
    SRLoggingLevel level,
    const std::string& data)
{
    Logger::get_instance().log_warning(context, level, data);
}

/*!
*   \brief Conditionally log a warning if the logging level is high enough
*   \param context Object containing the log context
*   \param level Minimum logging level for data to be logged
*   \param data Text of data to be logged
*/
inline void log_warning(
    const SRObject* context,
    SRLoggingLevel level,
    const std::string& data)
{
    context->log_warning(level, data);
}

/*!
*   \brief Conditionally log an error if the logging level is high enough
*   \param context Logging context (string to prefix the log entry with)
*   \param level Minimum logging level for data to be logged
*   \param data Text of data to be logged
*/
inline void log_error(
    const std::string& context,
    SRLoggingLevel level,
    const std::string& data)
{
    Logger::get_instance().log_error(context, level, data);
}

/*!
*   \brief Conditionally log an error if the logging level is high enough
*   \param context Object containing the log context
*   \param level Minimum logging level for data to be logged
*   \param data Text of data to be logged
*/
inline void log_error(
    const SRObject* context,
    SRLoggingLevel level,
    const std::string& data)
{
    context->log_error(level, data);
}


/////////////////////////////////////////////////////////
// Stack-based function logger

/*!
*   \brief The FunctionLogger class logs entry and exit of an API function.
*          The intended use is to create an instance of this class on the stack
*          inside each API point via the LOG_API_FUNCTION macro, below.
*/
class FunctionLogger {
    public:
        /*!
        *   \brief Logger constructor
        *   \param function_name The name of the function to track
        */
        FunctionLogger(const SRObject* context, const char* function_name)
            : _name(function_name), _context(context)
        {
            _context->log_data(
                LLDebug, "API Function " + _name + " called");
        }

        /*!
        *   \brief Logger destructor
        */
        ~FunctionLogger()
        {
            _context->log_data(
                LLDebug, "API Function " + _name + " exited");
        }
    private:
        /*!
        *   \brief The name of the current function
        */
        std::string _name;

        /*!
        *   \brief The logging context (enclosing object for the API function)
        */
        const SRObject* _context;
};

/*!
*   \brief Instantiate a FunctionLogger for the enclosing API function
*/
#define LOG_API_FUNCTION() \
    FunctionLogger ___function_logger___(this, __func__)


} // namespace SmartRedis
#endif // __cplusplus

#endif // SMARTREDIS_LOGGER_H
