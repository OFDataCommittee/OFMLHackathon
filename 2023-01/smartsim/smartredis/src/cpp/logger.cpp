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

// Convert a std::string to lower case
void str_to_lower(std::string& str)
{
    std::transform(
        str.begin(), str.end(), str.begin(),
        [](unsigned char c) { return std::tolower(c); }
    );
}

// Set up logging for the current client
void Logger::configure_logging()
{
    // Mark ourselves as initialized now
    _initialized = true;

    // Get the logfile
    get_config_string(_logfile, "SR_LOG_FILE", "", true);
    std::string requestedLogfile(_logfile);
    bool missingLogFile = _logfile.length() == 0;

    // Make sure it can be written to
    bool badLogFile = false;
    if (_logfile.length() > 0) {
        std::ofstream logstream;
        logstream.open(_logfile, std::ios_base::app);
        badLogFile = !logstream.good();

        // Switch to STDOUT if we can't write to the specified file
        if (badLogFile)
            _logfile = "";
    }

    // Get the logging level
    std::string level;
    get_config_string(level, "SR_LOG_LEVEL", "", true);
    bool missingLogLevel = level.length() == 0;
    bool badLogLevel = false;
    if (level.length() > 0) {
        str_to_lower(level);
        if (level.compare("quiet") == 0)
            _log_level = LLQuiet;
        else if (level.compare("info") == 0)
            _log_level = LLInfo;
        else if (level.compare("debug") == 0)
            _log_level = LLDebug;
        else if (level.compare("developer") == 0)
            _log_level = LLDeveloper;
        else {
            // Don't recognize the requested level; use default
            badLogLevel = true;
            _log_level = LLInfo;
        }
    }
    else {
        _log_level = LLInfo;
    }

    // Now that everything is configured, issue warning and
    // error messages. By deferring them, we may be able to
    // issue them to the customer-requested logfile instead
    // of to console, depending on what went wrong
    if (missingLogFile) {
        log_warning(
            "SmartRedis Library",
            LLInfo,
            "Environment variable SR_LOG_FILE is not set. "
            "Defaulting to stdout"
        );
    }
    if (missingLogLevel) {
        log_warning(
            "SmartRedis Library",
            LLInfo,
            "Environment variable SR_LOG_LEVEL is not set. "
            "Defaulting to INFO"
        );
    }
    if (badLogFile) {
        throw SRRuntimeException(
            "Cannot write to file: " + requestedLogfile);
    }
    if (badLogLevel) {
        throw SRRuntimeException(
            "Unrecognized logging level: " + level);
    }
}

// Conditionally log data if the logging level is high enough
void Logger::log_data(
    const std::string& context,
    SRLoggingLevel level,
    const std::string& data)
{
    // If we're not initialized, configure logging as a default
    // client. This can happen if a caller invokes a Dataset API point
    // without initializing a Client object
    if (!_initialized)
        configure_logging();

    // Silently ignore logging more verbose than requested
    if (level > _log_level)
        return;

    // Get the current timestamp
    auto t = std::time(NULL);
    auto tm = *std::localtime(&t);
    auto timestamp = std::put_time(&tm, "%H-%M-%S");

    // write the log data
    bool writingFile = (_logfile.length() > 0);
    std::ofstream logstream;
    std::ostream& log_target(writingFile ? logstream : std::cout);
    if (_logfile.length() > 0) {
        logstream.open(_logfile, std::ios_base::app);
        if (!logstream.good()) {
            // The logfile is no longer writable!
            // Switch to console and emit an error
            _logfile = "";
            log_error(
                "SmartRedis Library",
                LLInfo,
                "Logfile no longer writeable. Switching to console logging");
            // Re-log this message since the current attempt has failed
            log_data(context, level, data);
            // Bail -- we're done here
            return;
        }
    }
    log_target << context << "@" << timestamp << ":" << data
               << std::endl;
}
