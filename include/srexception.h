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

typedef enum {
    sr_ok        = 0, // No error
    sr_badalloc  = 1, // Memory allocation error
    sr_dberr     = 2, // Backend database error
    sr_internal  = 3, // Internal SmartRedis error
    sr_runtime   = 4, // Runtime error executing an operation
    sr_parameter = 5, // Bad parameter error
    sr_timeout   = 6, // Timeout error
    sr_invalid   = 7  // Uninitialized error variable
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
class SRException: public std::runtime_error
{
	// Inherit all the standard constructors
	using std::runtime_error::runtime_error;


	public:
	SRException(const char* what_arg, const char* file, int line)
	  : std::runtime_error(what_arg)
	{
		_loc = file;
	    _loc += ":" + std::to_string(line);
	}

	SRException(const std::string& what_arg, const char* file, int line)
	  : std::runtime_error(what_arg)
	{
		_loc = file;
	    _loc += ":" + std::to_string(line);
	}

	SRException(const SRException& other) noexcept
	  : std::runtime_error(other), _loc(other._loc)
	{
		// NOP
	}

	virtual SRError to_error_code() const noexcept {
		return sr_invalid;
	}

	virtual const char* what() const noexcept{
		std::string output(what());
		output += "\n" + _loc;
		return output.c_str();
	}

	virtual const char* what(bool hideLocation) const noexcept {
		return hideLocation ? std::runtime_error::what() : what();
	}

	virtual const char* where() const noexcept {
		return _loc.c_str();
	}

	protected:
	std::string _loc;
};

// Memory allocation error
class _SRBadAlloc: public SRException
{
	using SRException::SRException;

	virtual SRError to_error_code() const noexcept {
		return sr_badalloc;
	}
};

#define SRBadAlloc(txt) _SRBadAlloc(txt, __FILE__, __LINE__)

//  Back-end database error
class _SRDatabaseError: public SRException
{
	using SRException::SRException;

	virtual SRError to_error_code() const noexcept {
		return sr_dberr;
	}
};

#define SRDatabaseError(txt) _SRDatabaseError(txt, __FILE__, __LINE__)

// Runtime error
class _SRRuntimeError: public SRException
{
	using SRException::SRException;

	virtual SRError to_error_code() const noexcept {
		return sr_runtime;
	}
};

#define SRRuntimeError(txt) _SRRuntimeError(txt, __FILE__, __LINE__)

// Parameter error
class _SRParameterError: public SRException
{
	using SRException::SRException;

	virtual SRError to_error_code() const noexcept {
		return sr_parameter;
	}
};

#define SRParameterError(txt) _SRParameterError(txt, __FILE__, __LINE__)

#define SRRuntimeError(txt) _SRRuntimeError(txt, __FILE__, __LINE__)

// Timeout error
class _SRTimeoutError: public SRException
{
	using SRException::SRException;

	virtual SRError to_error_code() const noexcept {
		return sr_timeout;
	}
};

#define SRTimeoutError(txt) _SRTimeoutError(txt, __FILE__, __LINE__)

// Internal error
class _SRInternalError: public SRException
{
	using SRException::SRException;

	virtual SRError to_error_code() const noexcept {
		return sr_internal;
	}
};

#define SRInternalError(txt) _SRInternalError(txt, __FILE__, __LINE__)

// Store the last error encountered
extern "C"
void SRSetLastError(const SRException& last_error);

#endif // __cplusplus
#endif // SMARTREDIS_SREXCEPTION_H
