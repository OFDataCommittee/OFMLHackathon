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


#include "pyconfigoptions.h"
#include "srexception.h"

using namespace SmartRedis;

namespace py = pybind11;

// Decorator to standardize exception handling in PyBind Client API methods
template <class T>
auto pb_cfgopt_api(T&& cfgopt_api_func, const char* name)
{
  // we create a closure below
  auto decorated =
  [name, cfgopt_api_func = std::forward<T>(cfgopt_api_func)](auto&&... args)
  {
    try {
      return cfgopt_api_func(std::forward<decltype(args)>(args)...);
    }
    catch (Exception& e) {
        // exception is already prepared for caller
        throw;
    }
    catch (std::exception& e) {
        // should never happen
        throw SRInternalException(e.what());
    }
    catch (...) {
        // should never happen
        std::string msg(
            "A non-standard exception was encountered while executing ");
        msg += name;
        throw SRInternalException(msg);
    }
  };
  return decorated;
}

// Macro to invoke the decorator with a lambda function
#define MAKE_CFGOPT_API(stuff)\
    pb_cfgopt_api([&] { stuff }, __func__)()

PyConfigOptions::PyConfigOptions()
{
    MAKE_CFGOPT_API({
        _configoptions = NULL;
    });
}

PyConfigOptions::PyConfigOptions(ConfigOptions* configoptions)
{
    _configoptions = configoptions;
}

PyConfigOptions::~PyConfigOptions()
{
    MAKE_CFGOPT_API({
        if (_configoptions != NULL) {
            delete _configoptions;
            _configoptions = NULL;
        }
    });
}

ConfigOptions* PyConfigOptions::get() {
    return _configoptions;
}

// Instantiate ConfigOptions from environment variables
PyConfigOptions* PyConfigOptions::create_from_environment(
    const std::string& db_suffix)
{
    return MAKE_CFGOPT_API({
        auto cfgOpts = ConfigOptions::create_from_environment(db_suffix);
        ConfigOptions* pCfgOpts = cfgOpts.release();
        return new PyConfigOptions(pCfgOpts);
    });
}

// Retrieve the value of a numeric configuration option
// from the selected source
int64_t PyConfigOptions::get_integer_option(const std::string& option_name)
{
    return MAKE_CFGOPT_API({
        if (_configoptions == NULL) {
            throw SRRuntimeException(
                "Attempted to call get_integer_option "\
                "from a non-factory constructed ConfigOptions");
        }
        return _configoptions->get_integer_option(option_name);
    });
}

// Retrieve the value of a string configuration option
// from the selected source
std::string PyConfigOptions::get_string_option(const std::string& option_name)
{
    return MAKE_CFGOPT_API({
        if (_configoptions == NULL) {
            throw SRRuntimeException(
                "Attempted to call get_string_option "\
                "from a non-factory constructed ConfigOptions");
        }
        return _configoptions->get_string_option(option_name);
    });
}

// Check whether a configuration option is set in the
// selected source
bool PyConfigOptions::is_configured(const std::string& option_name)
{
    return MAKE_CFGOPT_API({
        if (_configoptions == NULL) {
            throw SRRuntimeException(
                "Attempted to call is_defined "\
                "from a non-factory constructed ConfigOptions");
        }
        return _configoptions->is_configured(option_name);
    });
}

// Override the value of a numeric configuration option
// in the selected source
void PyConfigOptions::override_integer_option(
    const std::string& option_name, int64_t value)
{
    MAKE_CFGOPT_API({
        if (_configoptions == NULL) {
            throw SRRuntimeException(
                "Attempted to call override_integer_option "\
                "from a non-factory constructed ConfigOptions");
        }
        _configoptions->override_integer_option(option_name, value);
    });
}

// Override the value of a string configuration option
// in the selected source
void PyConfigOptions::override_string_option(
    const std::string& option_name, const std::string& value)
{
    MAKE_CFGOPT_API({
        if (_configoptions == NULL) {
            throw SRRuntimeException(
                "Attempted to call override_string_option "\
                "from a non-factory constructed ConfigOptions");
        }
        _configoptions->override_string_option(option_name, value);
    });
}

// EOF

