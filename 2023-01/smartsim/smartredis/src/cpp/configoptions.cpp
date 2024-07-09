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

#include <string_view>
#include <algorithm>
#include "configoptions.h"
#include "srexception.h"
#include "logger.h"
#include "utility.h"

using namespace SmartRedis;

// ConfigOptions constructor
ConfigOptions::ConfigOptions(
    cfgSrc source,
    const std::string& string)
    : _source(source), _string(string), _lazy(source == cs_envt),
      _log_context(NULL)
{
    // Read in options if needed
    if (!_lazy) {
        _populate_options();
    }
}

// Deep copy a ConfigOptions object
ConfigOptions* ConfigOptions::clone()
{
    ConfigOptions* result = new ConfigOptions(_source, _string);
    result->_log_context = _log_context;
    result->_int_options = _int_options;
    result->_string_options = _string_options;
    return result;
}

// ConfigOptions destructor
ConfigOptions::~ConfigOptions()
{
    // Nuke each string from our stash
    auto nuke = [](char* buf) { delete buf; };
    std::for_each(_string_buffer_stash.begin(), _string_buffer_stash.end(), nuke);
    _string_buffer_stash.clear();
}

// Instantiate ConfigOptions, getting selections from environment variables
std::unique_ptr<ConfigOptions> ConfigOptions::create_from_environment(
    const std::string& db_suffix)
{
    // NOTE: We can't use std::make_unique<> here because our constructor
    // is private
    return std::unique_ptr<ConfigOptions>(
        new ConfigOptions(cs_envt, db_suffix));
}

// Instantiate ConfigOptions, getting selections from environment variables
std::unique_ptr<ConfigOptions> ConfigOptions::create_from_environment(
    const char* db_suffix)
{
    std::string str_suffix(db_suffix != NULL ? db_suffix : "");
    return create_from_environment(str_suffix);
}

// Retrieve the value of a numeric configuration option
int64_t ConfigOptions::get_integer_option(const std::string& option_name)
{
    // If we already have the value, return it
    auto search = _int_options.find(option_name);
    if (search != _int_options.end())
        return _int_options[option_name];

    // If we're doing lazy evaluation of option names, fetch the value
    int64_t default_value = -1;
    int64_t result = default_value;
    if (_lazy) {
        int temp = 0;
        get_config_integer(
            temp, _suffixed(option_name), default_value, throw_on_absent);
        result = (int64_t)temp;
    }

    // Store the final value before we exit
    _int_options.insert({option_name, result});
    return result;
}

// Retrieve the value of a string configuration option
std::string ConfigOptions::get_string_option(const std::string& option_name)
{
    // If we already have the value, return it
    auto search = _string_options.find(option_name);
    if (search != _string_options.end())
        return _string_options[option_name];

    // If we're doing lazy evaluation of option names, fetch the value
    std::string default_value("");
    std::string result(default_value);
    if (_lazy) {
        get_config_string(
            result, _suffixed(option_name), default_value, throw_on_absent);
    }

    // Store the final value before we exit
    _string_options.insert({option_name, result});
    return result;
}

// Resolve the value of a numeric configuration option
int64_t ConfigOptions::_resolve_integer_option(
    const std::string& option_name, int64_t default_value)
{
    // If we already have the value, return it
    auto search = _int_options.find(option_name);
    if (search != _int_options.end())
        return _int_options[option_name];

    // If we're doing lazy evaluation of option names, fetch the value
    int64_t result = default_value;
    if (_lazy) {
        int temp = 0;
        get_config_integer(temp, _suffixed(option_name), default_value);
        result = (int64_t)temp;
    }

    // Store the final value before we exit
    _int_options.insert({option_name, result});
    return result;
}

// Resolve the value of a string configuration option
std::string ConfigOptions::_resolve_string_option(
    const std::string& option_name, const std::string& default_value)
{
    // If we already have the value, return it
    auto search = _string_options.find(option_name);
    if (search != _string_options.end())
        return _string_options[option_name];

    // If we're doing lazy evaluation of option names, fetch the value
    std::string result(default_value);
    if (_lazy) {
        get_config_string(result, _suffixed(option_name), default_value);
    }

    // Store the final value before we exit
    _string_options.insert({option_name, result});
    return result;
}

// Check whether a configuration option is set in the selected source
bool ConfigOptions::is_configured(const std::string& option_name)
{
    // Check each map in turn
    if (_int_options.find(option_name) != _int_options.end())
        return true;
    if (_string_options.find(option_name) != _string_options.end())
        return true;

    // Check to see if the value is available and we just haven't
    // seen it yet
    if (_lazy) {
        std::string suffixed = _suffixed(option_name);
        char* environment_string = std::getenv(suffixed.c_str());
        return NULL != environment_string;
    }

    // Not found
    return false;
}

// Override the value of a numeric configuration option
void ConfigOptions::override_integer_option(
    const std::string& option_name, int64_t value)
{
    _int_options.insert_or_assign(option_name, value);
}

// Override the value of a string configuration option
void ConfigOptions::override_string_option(
    const std::string& option_name, const std::string& value)
{
    _string_options.insert_or_assign(option_name, value);
}

// Process option data from a fixed source
void ConfigOptions::_populate_options()
{
    throw SRRuntimeException(
        "Sources other than environment variables "
        "are not currently supported"
    );
}

// Apply a suffix to a option_name if the source is environment
// variables and the suffix is nonempty
std::string ConfigOptions::_suffixed(const std::string& option_name)
{
    // Sanity check
    if ("" == option_name) {
        throw SRKeyException(
            "Invalid empty environment variable name detected");
    }
    std::string result(option_name);
    if (_source == cs_envt && _string != "")
        result = option_name + + "_" + _string;
    return result;
}

// Clear a configuration option from the cache
void ConfigOptions::_clear_option_from_cache(const std::string& option_name)
{
    _int_options.erase(option_name);
    _string_options.erase(option_name);
}
