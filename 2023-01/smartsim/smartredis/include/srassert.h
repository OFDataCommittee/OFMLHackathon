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

#ifndef SMARTREDIS_SRASSERT_H
#define SMARTREDIS_SRASSERT_H

#include "srexception.h"

using namespace SmartRedis;

///@file

/*!
*   \brief  Throw a SmartRedis::RuntimeException. Written as a boolean function in order
*           that it may be logically joined with an assertion (leveraging short-circuiting)
*           and not impact code coverage statistics for adding assertions
*   \param txt Message for the RuntimeException
*   \param file Source file for where the RuntimeException should report as thrown from
*   \param line Line number for where the RuntimeException should report as thrown from
*   \returns This routine always throws an exception
*/
inline bool throw_runtime_exception(const std::string &txt, const char *file, int line)
{
    throw RuntimeException(txt, file, line);
}

/*!
*   \brief Assert that a condition is true or throw a RuntimeException
*/
#define SR_ASSERT(condition) \
    ((condition) || throw_runtime_exception(std::string("Assertion failed!") + #condition, __FILE__, __LINE__))


/*!
*   \brief  Throw a SmartRedis::ParameterException. Written as a boolean function in order
*           that it may be logically joined with an assertion (leveraging short-circuiting)
*           and not impact code coverage statistics for adding parameter checks
*   \param txt Message for the ParameterException
*   \param file Source file for where the RuntimeException should report as thrown from
*   \param line Line number for where the RuntimeException should report as thrown from
*   \returns This routine always throws an exception
*/
inline bool throw_param_exception(const std::string &txt, const char *file, int line)
{
    throw ParameterException(txt, file, line);
}

/*!
*   \brief Validate a parameter list (via a boolean condition) or throw a RuntimeException
*/
#define SR_CHECK_PARAMS(condition) \
    ((condition) || throw_param_exception(std::string("Assertion failed!") + #condition, __FILE__, __LINE__))

#endif // SMARTREDIS_SRASSERT_H
