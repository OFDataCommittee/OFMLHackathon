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


#include "pylogcontext.h"
#include "logcontext.h"
#include "srexception.h"

using namespace SmartRedis;

namespace py = pybind11;

// Decorator to standardize exception handling in PyBind LogContext API methods
template <class T>
auto pb_logcontext_api(T&& logcontext_api_func, const char* name)
{
  // we create a closure below
  auto decorated =
  [name, logcontext_api_func = std::forward<T>(logcontext_api_func)](auto&&... args)
  {
    try {
      return logcontext_api_func(std::forward<decltype(args)>(args)...);
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
#define MAKE_LOGCONTEXT_API(stuff)\
    pb_logcontext_api([&] { stuff }, __func__)()



PyLogContext::PyLogContext(const std::string& context)
    : PySRObject(context)
{
    MAKE_LOGCONTEXT_API({
        _logcontext = new LogContext(context);
    });
}

PyLogContext::PyLogContext(LogContext* logcontext)
    : PySRObject(logcontext->get_context())
{
    MAKE_LOGCONTEXT_API({
        _logcontext = logcontext;
    });
}

PyLogContext::~PyLogContext()
{
    MAKE_LOGCONTEXT_API({
        if (_logcontext != NULL) {
            delete _logcontext;
            _logcontext = NULL;
        }
    });
}
