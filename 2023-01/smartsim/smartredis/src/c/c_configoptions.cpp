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

#include <iostream>
#include <cstring>
#include "srassert.h"
#include "srexception.h"
#include "configoptions.h"

using namespace SmartRedis;

// Decorator to standardize exception handling in C ConfigOptions API methods
template <class T>
auto c_cfgopt_api(T&& cfgopt_api_func, const char* name)
{
  // we create a closure below
  auto decorated = [name, cfgopt_api_func =
    std::forward<T>(cfgopt_api_func)](auto&&... args)
  {
    SRError result = SRNoError;
    try {
      cfgopt_api_func(std::forward<decltype(args)>(args)...);
    }
    catch (const Exception& e) {
      SRSetLastError(e);
      result = e.to_error_code();
    }
    catch (...) {
      std::string msg(
          "A non-standard exception was encountered while executing ");
      msg += name;
      SRSetLastError(SRInternalException(msg));
      result = SRInternalError;
    }
    return result;
  };
  return decorated;
}

// Macro to invoke the decorator with a lambda function
#define MAKE_CFGOPT_API(stuff)\
    c_cfgopt_api([&] { stuff }, __func__)()

// Instantiate ConfigOptions from environment variables
extern "C"
SRError create_configoptions_from_environment(
    const char* db_suffix,
    const size_t db_suffix_length,
    void** new_configoptions)
{
  return MAKE_CFGOPT_API({
    try {
      // Sanity check params
      SR_CHECK_PARAMS(db_suffix != NULL && new_configoptions != NULL);

      std::string db_suffix_str(db_suffix, db_suffix_length);

      auto cfgOpts = ConfigOptions::create_from_environment(db_suffix_str);
      ConfigOptions* pCfgOpts = cfgOpts.release();
      *new_configoptions = reinterpret_cast<void*>(pCfgOpts);
    }
    catch (const std::bad_alloc& e) {
      throw SRBadAllocException("config options allocation");
    }
  });
}

// Retrieve the value of a numeric configuration option
extern "C"
SRError get_integer_option(
    void* c_cfgopts,
    const char* option_name,
    size_t option_name_len,
    int64_t* option_result)
{
  return MAKE_CFGOPT_API({
    // Sanity check params
    SR_CHECK_PARAMS(c_cfgopts != NULL && option_name != NULL &&
      option_name_len > 0 && option_result != NULL);

    std::string option_name_str(option_name, option_name_len);
    ConfigOptions* co = reinterpret_cast<ConfigOptions*>(c_cfgopts);

    *option_result = co->get_integer_option(option_name_str);
  });
}

// Retrieve the value of a string configuration option
extern "C"
SRError get_string_option(
    void* c_cfgopts,
    const char* option_name,
    size_t option_name_len,
    char** option_result,
    size_t* option_result_len)
{
  return MAKE_CFGOPT_API({
    // Sanity check params
    SR_CHECK_PARAMS(c_cfgopts != NULL && option_name != NULL &&
      option_name_len > 0 && option_result != NULL &&
      option_result_len != NULL);

    std::string option_name_str(option_name, option_name_len);
    ConfigOptions* co = reinterpret_cast<ConfigOptions*>(c_cfgopts);

    // Set up an empty string as the result in case something goes wrong
    *option_result = NULL;
    *option_result = 0;

    std::string option_result_str = co->get_string_option(option_name_str);

    *option_result_len = option_result_str.length();
    *option_result = new char[*option_result_len + 1];
    strncpy(*option_result, option_result_str.c_str(), *option_result_len);

     // Save the pointer to this buffer so we can clean it up later
    co->_add_string_buffer(*option_result);
  });
}

// Check whether a configuration option is set
extern "C"
SRError is_configured(
    void* c_cfgopts,
    const char* option_name,
    size_t option_name_len,
    bool* cfg_result)
{
  return MAKE_CFGOPT_API({
    // Sanity check params
    SR_CHECK_PARAMS(c_cfgopts != NULL && option_name != NULL && cfg_result != NULL);

    std::string option_name_str(option_name, option_name_len);
    ConfigOptions* co = reinterpret_cast<ConfigOptions*>(c_cfgopts);

    *cfg_result = co->is_configured(option_name_str);
  });
}

// Override the value of a numeric configuration option
extern "C"
SRError override_integer_option(
    void* c_cfgopts,
    const char* option_name,
    size_t option_name_len,
    int64_t value)
{
  return MAKE_CFGOPT_API({
    // Sanity check params
    SR_CHECK_PARAMS(c_cfgopts != NULL && option_name != NULL &&
      option_name_len > 0);

    std::string option_name_str(option_name, option_name_len);
    ConfigOptions* co = reinterpret_cast<ConfigOptions*>(c_cfgopts);

    co->override_integer_option(option_name_str, value);
  });
}

// Override the value of a string configuration option
extern "C"
SRError override_string_option(
    void* c_cfgopts,
    const char* option_name,
    size_t option_name_len,
    const char* value,
    size_t value_len)
{
  return MAKE_CFGOPT_API({
    // Sanity check params
    SR_CHECK_PARAMS(c_cfgopts != NULL && option_name != NULL &&
      option_name_len > 0 && value != NULL);

    std::string option_name_str(option_name, option_name_len);
    std::string value_str(value, value_len);
    ConfigOptions* co = reinterpret_cast<ConfigOptions*>(c_cfgopts);

    co->override_string_option(option_name_str, value_str);
  });
}
