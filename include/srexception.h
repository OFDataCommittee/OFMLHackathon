/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Hewlett Packard Enterprise
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

#ifndef SMARTREDIS_SRException_H
#define SMARTREDIS_SRException_H

#include <stdio.h>
#include <stdlib.h>

///@file

/*!
*   \brief  SRError lists possible errors encountered in SmartRedis.
*/
typedef enum {
    SRNoError        = 0, // No error
    SRBadAllocError  = 1, // Memory allocation error
    SRDatabaseError  = 2, // Backend database error
    SRInternalError  = 3, // Internal SmartRedis error
    SRRuntimeError   = 4, // Runtime error executing an operation
    SRParameterError = 5, // Bad parameter error
    SRTimeoutError   = 6, // Timeout error
    SRKeyError       = 7, // Key error
    SRInvalidError   = 8, // Uninitialized error variable
    SRTypeError      = 9  // Type mismatch
} SRError;


/*!
*   \brief Return the last error encountered
*   \return The text data for the last error encountered
*/
#ifdef __cplusplus
extern "C"
#endif
const char* SRGetLastError();

#ifdef __cplusplus
#include <string>
namespace SmartRedis {

/*!
*   \brief  Smart error: custom exception class for the SmartRedis library
*/
class Exception: public std::exception
{
    public:
    Exception(const char* what_arg)
      : _msg(what_arg)
    {
        // NOP
    }

    Exception(const char* what_arg, const char* file, int line)
      : _msg(what_arg), _loc(file + std::string(":") + std::to_string(line))
    {
        // NOP
    }

    Exception(const std::string& what_arg, const char* file, int line)
      : _msg(what_arg), _loc(file + std::string(":") + std::to_string(line))
    {
        // NOP
    }

    Exception(const Exception& other) noexcept
      : _msg(other._msg), _loc(other._loc)
    {
        // NOP
    }

    Exception(const std::exception& other) noexcept
      : _msg(other.what())
    {
        // NOP
    }

    Exception& operator=(const Exception &) = default;
    Exception(Exception &&) = default;
    Exception& operator=(Exception &&) = default;
    virtual ~Exception() override = default;

    virtual SRError to_error_code() const noexcept {
        return SRInvalidError;
    }

    virtual const char* what() const noexcept{
        return _msg.c_str();
    }

    virtual const char* where() const noexcept {
        return _loc.c_str();
    }

    protected:
    std::string _msg;
    std::string _loc;
};


/*!
*   \brief  Memory allocation exception for SmartRedis
*/
class BadAllocException: public Exception
{
    public:
    using Exception::Exception;

    virtual SRError to_error_code() const noexcept {
        return SRBadAllocError;
    }
};

#define SRBadAllocException(txt) BadAllocException(txt, __FILE__, __LINE__)


/*!
*   \brief  Back-end database exception for SmartRedis
*/
class DatabaseException: public Exception
{
    public:
    using Exception::Exception;

    virtual SRError to_error_code() const noexcept {
        return SRDatabaseError;
    }
};

#define SRDatabaseException(txt) DatabaseException(txt, __FILE__, __LINE__)


/*!
*   \brief  Runtime exception for SmartRedis
*/
class RuntimeException: public Exception
{
    public:
    using Exception::Exception;

    virtual SRError to_error_code() const noexcept {
        return SRRuntimeError;
    }
};

#define SRRuntimeException(txt) RuntimeException(txt, __FILE__, __LINE__)


/*!
*   \brief  Parameter exception for SmartRedis
*/
class ParameterException: public Exception
{
    public:
    using Exception::Exception;

    virtual SRError to_error_code() const noexcept {
        return SRParameterError;
    }
};

#define SRParameterException(txt) ParameterException(txt, __FILE__, __LINE__)


/*!
*   \brief  Timeout exception for SmartRedis
*/
class TimeoutException: public Exception
{
    public:
    using Exception::Exception;

    virtual SRError to_error_code() const noexcept {
        return SRTimeoutError;
    }
};

#define SRTimeoutException(txt) TimeoutException(txt, __FILE__, __LINE__)


/*!
*   \brief  Internal exception for SmartRedis
*/
class InternalException: public Exception
{
    public:
    using Exception::Exception;

    virtual SRError to_error_code() const noexcept {
        return SRInternalError;
    }
};

#define SRInternalException(txt) InternalException(txt, __FILE__, __LINE__)


/*!
*   \brief  Key exception for SmartRedis
*/
class KeyException: public Exception
{
    public:
    using Exception::Exception;

    virtual SRError to_error_code() const noexcept {
        return SRKeyError;
    }
};

#define SRKeyException(txt) KeyException(txt, __FILE__, __LINE__)

/*!
*   \brief  Type mismatch exception for SmartRedis
*/
class TypeException: public Exception
{
    public:
    using Exception::Exception;

    virtual SRError to_error_code() const noexcept {
        return SRTypeError;
    }
};

#define SRTypeException(txt) TypeException(txt, __FILE__, __LINE__)

/*!
*   \brief  Store the last error encountered
*/
extern "C"
void SRSetLastError(const Exception& last_error);

} // namespace SmartRedis

#endif // __cplusplus
#endif // SMARTREDIS_SRException_H
