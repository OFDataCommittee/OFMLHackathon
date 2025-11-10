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

#include "c_logcontext.h"
#include "srexception.h"
#include "srassert.h"

using namespace SmartRedis;

// Decorator to standardize exception handling in C LogContext API methods
template <class T>
auto c_logcontext_api(T&& logcontext_api_func, const char* name)
{
  // we create a closure below
  auto decorated = [name, logcontext_api_func
    = std::forward<T>(logcontext_api_func)](auto&&... args)
  {
    SRError result = SRNoError;
    try {
      logcontext_api_func(std::forward<decltype(args)>(args)...);
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
#define MAKE_LOGCONTEXT_API(stuff)\
    c_logcontext_api([&] { stuff }, __func__)()


// Create a new LogContext
extern "C" SRError SmartRedisCLogContext(
  const char* context, const size_t context_length, void** new_logcontext)
{
  return MAKE_LOGCONTEXT_API({
    // Sanity check params
    SR_CHECK_PARAMS(context != NULL && new_logcontext != NULL);

    try {
      std::string context_str(context, context_length);
      *new_logcontext = NULL;
      LogContext* logcontext = new LogContext(context_str);
      *new_logcontext = reinterpret_cast<void*>(logcontext);
    }
    catch (const std::bad_alloc& e) {
      SRSetLastError(SRBadAllocException("logcontext allocation"));
    }
  });
}

// Deallocate a LogContext
extern "C" SRError DeallocateLogContext(void** logcontext)
{
  return MAKE_LOGCONTEXT_API({
    // Sanity check params
    SR_CHECK_PARAMS(logcontext != NULL);

    LogContext* lc = reinterpret_cast<LogContext*>(*logcontext);
    delete lc;
    *logcontext = NULL;
  });
}
