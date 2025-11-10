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


#include "pysrobject.h"
#include "srobject.h"
#include "srexception.h"
#include "logger.h"

using namespace SmartRedis;

namespace py = pybind11;


// Decorator to standardize exception handling in PyBind SRObject API methods
template <class T>
auto pb_srobject_api(T&& srobject_api_func, const char* name)
{
  // we create a closure below
  auto decorated =
  [name, srobject_api_func = std::forward<T>(srobject_api_func)](auto&&... args)
  {
    try {
      return srobject_api_func(std::forward<decltype(args)>(args)...);
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
#define MAKE_SROBJECT_API(stuff)\
    pb_srobject_api([&] { stuff }, __func__)()


PySRObject::PySRObject(const std::string& context)
{
    MAKE_SROBJECT_API({
        _srobject = new SRObject(context);
    });
}

PySRObject::PySRObject(SRObject* srobject)
{
    MAKE_SROBJECT_API({
        _srobject = srobject;
    });
}

PySRObject::~PySRObject()
{
    MAKE_SROBJECT_API({
        if (_srobject != NULL) {
            delete _srobject;
            _srobject = NULL;
        }
    });
}

void PySRObject::log_data(
    SRLoggingLevel level, const std::string& data) const
{
    MAKE_SROBJECT_API({
        _srobject->log_data(level, data);
    });
}

void PySRObject::log_warning(
    SRLoggingLevel level, const std::string& data) const
{
    MAKE_SROBJECT_API({
        _srobject->log_warning(level, data);
    });
}

void PySRObject::log_error(
    SRLoggingLevel level, const std::string& data) const
{
    MAKE_SROBJECT_API({
        _srobject->log_error(level, data);
    });
}
