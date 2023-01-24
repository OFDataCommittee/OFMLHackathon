/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2022, Hewlett Packard Enterprise
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
    /*!
    *   \brief Exception default constructor
    *   \param what_arg The message for the exception
    */
    Exception(const char* what_arg)
      : _msg(what_arg)
    {
        // NOP
    }

    /*!
    *   \brief Exception constructor with location information
    *   \param what_arg The message for the exception
    *   \param file The source file from which the exception was thrown
    *   \param line The line number from which the exception was thrown
    */
    Exception(const char* what_arg, const char* file, int line)
      : _msg(what_arg), _loc(file + std::string(":") + std::to_string(line))
    {
        // NOP
    }

    /*!
    *   \brief Exception constructor with location information
    *   \param what_arg The message for the exception
    *   \param file The source file from which the exception was thrown
    *   \param line The line number from which the exception was thrown
    */
    Exception(const std::string& what_arg, const char* file, int line)
      : _msg(what_arg), _loc(file + std::string(":") + std::to_string(line))
    {
        // NOP
    }

    /*!
    *   \brief Exception copy constructor
    *   \param other Exception to copy
    */
    Exception(const Exception& other) noexcept
      : _msg(other._msg), _loc(other._loc)
    {
        // NOP
    }

    /*!
    *   \brief Exception copy constructor for std::exception
    *   \param other Standard exception to copy
    */
    Exception(const std::exception& other) noexcept
      : _msg(other.what())
    {
        // NOP
    }

    /*!
    *   \brief Exception assignment operator
    *   \param other Exception to assign
    */
    Exception& operator=(const Exception& other) = default;

    /*!
    *   \brief Exception move constructor
    *   \param other Exception to move
    */
    Exception(Exception&& other) = default;

    /*!
    *   \brief Exception move assignment operator
    *   \param other Exception to move
    */
    Exception& operator=(Exception&& other) = default;

    /*!
    *   \brief Exception destructor
    */
    virtual ~Exception() override = default;

    /*!
    *   \brief Retrieve an error code matching the exception type
    *   \returns Error code corresponding to the exception type
    */
    virtual SRError to_error_code() const noexcept {
        return SRInvalidError;
    }

    /*!
    *   \brief Retrieve the message for an exception
    *   \returns String of message data
    */
    const char* what() const noexcept override {
        return _msg.c_str();
    }

    /*!
    *   \brief Retrieve the location data for an exception
    *   \returns String with location data
    */
    virtual const char* where() const noexcept {
        return _loc.c_str();
    }

    protected:

    /*!
    *   \brief  The exception message text
    */
    std::string _msg;

    /*!
    *   \brief  The exception location text
    */
    std::string _loc;
};


/*!
*   \brief  Memory allocation exception for SmartRedis
*/
class BadAllocException: public Exception
{
    public:
    using Exception::Exception;

    /*!
    *   \brief Retrieve an error code matching the exception type
    *   \returns Error code corresponding to the exception type
    */
    SRError to_error_code() const noexcept override {
        return SRBadAllocError;
    }
};

/*!
*   \brief Instantiate a BadAllocException with message
*/
#define SRBadAllocException(txt) BadAllocException(txt, __FILE__, __LINE__)


/*!
*   \brief  Back-end database exception for SmartRedis
*/
class DatabaseException: public Exception
{
    public:
    using Exception::Exception;

    /*!
    *   \brief Retrieve an error code matching the exception type
    *   \returns Error code corresponding to the exception type
    */
    SRError to_error_code() const noexcept override {
        return SRDatabaseError;
    }
};

/*!
*   \brief Instantiate a DatabaseException with message
*/
#define SRDatabaseException(txt) DatabaseException(txt, __FILE__, __LINE__)


/*!
*   \brief  Runtime exception for SmartRedis
*/
class RuntimeException: public Exception
{
    public:
    using Exception::Exception;

    /*!
    *   \brief Retrieve an error code matching the exception type
    *   \returns Error code corresponding to the exception type
    */
    SRError to_error_code() const noexcept override {
        return SRRuntimeError;
    }
};

/*!
*   \brief Instantiate a RuntimeException with message
\*/
#define SRRuntimeException(txt) RuntimeException(txt, __FILE__, __LINE__)


/*!
*   \brief  Parameter exception for SmartRedis
*/
class ParameterException: public Exception
{
    public:
    using Exception::Exception;

    /*!
    *   \brief Retrieve an error code matching the exception type
    *   \returns Error code corresponding to the exception type
    */
    SRError to_error_code() const noexcept override {
        return SRParameterError;
    }
};

/*!
*   \brief Instantiate a ParameterException with message
*/
#define SRParameterException(txt) ParameterException(txt, __FILE__, __LINE__)


/*!
*   \brief  Timeout exception for SmartRedis
*/
class TimeoutException: public Exception
{
    public:
    using Exception::Exception;

    /*!
    *   \brief Retrieve an error code matching the exception type
    *   \returns Error code corresponding to the exception type
    */
    SRError to_error_code() const noexcept override {
        return SRTimeoutError;
    }
};

/*!
*   \brief Instantiate a TimeoutException with message
*/
#define SRTimeoutException(txt) TimeoutException(txt, __FILE__, __LINE__)


/*!
*   \brief  Internal exception for SmartRedis
*/
class InternalException: public Exception
{
    public:
    using Exception::Exception;

    /*!
    *   \brief Retrieve an error code matching the exception type
    *   \returns Error code corresponding to the exception type
    */
    SRError to_error_code() const noexcept override {
        return SRInternalError;
    }
};

/*!
*   \brief Instantiate a InternalException with message
*/
#define SRInternalException(txt) InternalException(txt, __FILE__, __LINE__)


/*!
*   \brief  Key exception for SmartRedis
*/
class KeyException: public Exception
{
    public:
    using Exception::Exception;

    /*!
    *   \brief Retrieve an error code matching the exception type
    *   \returns Error code corresponding to the exception type
    */
    SRError to_error_code() const noexcept override {
        return SRKeyError;
    }
};

/*!
*   \brief Instantiate a KeyException with message
*/
#define SRKeyException(txt) KeyException(txt, __FILE__, __LINE__)

/*!
*   \brief  Type mismatch exception for SmartRedis
*/
class TypeException: public Exception
{
    public:
    using Exception::Exception;

    /*!
    *   \brief Retrieve an error code matching the exception type
    *   \returns Error code corresponding to the exception type
    */
    SRError to_error_code() const noexcept override {
        return SRTypeError;
    }
};

/*!
*   \brief Instantiate a TypeException with message
*/
#define SRTypeException(txt) TypeException(txt, __FILE__, __LINE__)

/*!
*   \brief Store the last error encountered in a global variable. Not
*          currently thread-safe
*   \param last_error Exception to be stored as the last error encountered
*/
extern "C"
void SRSetLastError(const Exception& last_error);

} // namespace SmartRedis

#endif // __cplusplus
#endif // SMARTREDIS_SRException_H
