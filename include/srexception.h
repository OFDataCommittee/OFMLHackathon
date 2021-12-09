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

#if !defined(SMARTREDIS_SREXCEPTION_H)
#define SMARTREDIS_SREXCEPTION_H

#include <stdio.h>
#include <stdlib.h>

typedef enum {
    SRNoError        = 0, // No error
    SRBadAllocError  = 1, // Memory allocation error
    SRDatabaseError  = 2, // Backend database error
    SRInternalError  = 3, // Internal SmartRedis error
    SRRuntimeError   = 4, // Runtime error executing an operation
    SRParameterError = 5, // Bad parameter error
    SRTimeoutError   = 6, // Timeout error
    SRInvalidError   = 7  // Uninitialized error variable
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

// Smart error: custom error class for the SmartRedis library
class SRException: public std::exception
{
	// Inherit all the standard constructors
	// using std::exception::exception;


	public:
	SRException(const char* what_arg)
	  : _msg(what_arg)
	{
	    // NOP
	}

	SRException(const char* what_arg, const char* file, int line)
	  : _msg(what_arg), _loc(file + std::string(":") + std::to_string(line))
	{
	    // NOP
	}

	SRException(const std::string& what_arg, const char* file, int line)
	  : _msg(what_arg), _loc(file + std::string(":") + std::to_string(line))
	{
	    // NOP
	}

	SRException(const SRException& other) noexcept
	  : _msg(other._msg), _loc(other._loc)
	{
		// NOP
	}

	SRException(const std::exception& other) noexcept
	  : _msg(other.what())
	{
		// NOP
	}

    SRException& operator=(const SRException &) = default;
    SRException(SRException &&) = default;
    SRException& operator=(SRException &&) = default;
    virtual ~SRException() override = default;

	virtual SRError to_error_code() const noexcept {
		return SRInvalidError;
	}

	virtual const char* what() const noexcept{
		std::string output = _msg + "\n" + _loc;
		return output.c_str();
	}

	virtual const char* what(bool hideLocation) const noexcept {
		return hideLocation ? _msg.c_str() : what();
	}

	virtual const char* where() const noexcept {
		return _loc.c_str();
	}

	protected:
	std::string _msg;
	std::string _loc;
};

//////////////////////////////////////////////////
// Memory allocation exception
class _SRBadAllocException: public SRException
{
	using SRException::SRException;

	virtual SRError to_error_code() const noexcept {
		return SRBadAllocError;
	}
};

#define SRBadAllocException(txt) _SRBadAllocException(txt, __FILE__, __LINE__)

//////////////////////////////////////////////////
//  Back-end database exception
class _SRDatabaseException: public SRException
{
	using SRException::SRException;

	virtual SRError to_error_code() const noexcept {
		return SRDatabaseError;
	}
};

#define SRDatabaseException(txt) _SRDatabaseException(txt, __FILE__, __LINE__)

//////////////////////////////////////////////////
// Runtime exception
class _SRRuntimeException: public SRException
{
	using SRException::SRException;

	virtual SRError to_error_code() const noexcept {
		return SRRuntimeError;
	}
};

#define SRRuntimeException(txt) _SRRuntimeException(txt, __FILE__, __LINE__)

//////////////////////////////////////////////////
// Parameter exception
class _SRParameterException: public SRException
{
	using SRException::SRException;

	virtual SRError to_error_code() const noexcept {
		return SRParameterError;
	}
};

#define SRParameterException(txt) _SRParameterException(txt, __FILE__, __LINE__)

//////////////////////////////////////////////////
// Timeout exception
class _SRTimeoutException: public SRException
{
	using SRException::SRException;

	virtual SRError to_error_code() const noexcept {
		return SRTimeoutError;
	}
};

#define SRTimeoutException(txt) _SRTimeoutException(txt, __FILE__, __LINE__)

//////////////////////////////////////////////////
// Internal exception
class _SRInternalException: public SRException
{
	using SRException::SRException;

	virtual SRError to_error_code() const noexcept {
		return SRInternalError;
	}
};

#define SRInternalException(txt) _SRInternalException(txt, __FILE__, __LINE__)

//////////////////////////////////////////////////
// Store the last error encountered
extern "C"
void SRSetLastError(const SRException& last_error);

#endif // __cplusplus
#endif // SMARTREDIS_SREXCEPTION_H
