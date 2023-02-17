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

PySRObject::PySRObject(const std::string& context)
{
    _srobject = NULL;
    try {
        _srobject = new SRObject(context);
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
        throw SRInternalException("A non-standard exception was encountered "\
                                  "during dataset construction.");
    }
}

PySRObject::PySRObject(SRObject* srobject)
{
    _srobject = srobject;
}

PySRObject::~PySRObject()
{
    if (_srobject != NULL) {
        delete _srobject;
        _srobject = NULL;
    }
}

SRObject* PySRObject::get() {
    return _srobject;
}

void PySRObject::log_data(
    SRLoggingLevel level, const std::string& data) const
{
    try {
        _srobject->log_data(level, data);
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
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing log_data.");
    }
}

void PySRObject::log_warning(
    SRLoggingLevel level, const std::string& data) const
{
    try {
        _srobject->log_warning(level, data);
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
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing log_warning.");
    }
}

void PySRObject::log_error(
    SRLoggingLevel level, const std::string& data) const
{
    try {
        _srobject->log_error(level, data);
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
        throw SRInternalException("A non-standard exception was encountered "\
                                  "while executing log_error.");
    }
}
