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

#ifndef SMARTREDIS_PYLOGCONTEXT_H
#define SMARTREDIS_PYLOGCONTEXT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "logcontext.h"
#include "pysrobject.h"

///@file

namespace py = pybind11;

namespace SmartRedis {

/*!
*   \brief The PyLogContext class is a wrapper around the
           C++ LogContext class.
*/
class PyLogContext : public PySRObject
{
    public:

        /*!
        *   \brief PyLogContext constructor
        *   \param context The context to use for log messages
        */
        PyLogContext(const std::string& context);

        /*!
        *   \brief PyLogContext constructor from a
        *          SmartRedis::LogContext object
        *   \param logcontext A SmartRedis::LogContext pointer to
        *                  a SmartRedis::LogContext allocated on
        *                  the heap.  The SmartRedis::LogContext
        *                  will be deleted upton PyLogContext
        *                  deletion.
        */
        PyLogContext(LogContext* logcontext);

        /*!
        *   \brief PyLogContext destructor
        */
        virtual ~PyLogContext();

        /*!
        *   \brief Retrieve a pointer to the underlying
        *          SmartRedis::LogContext object
        *   \returns LogContext pointer within PyLogContext
        */
        LogContext* get();

    private:

        LogContext* _logcontext;
};

} // namespace SmartRedis

#endif // SMARTREDIS_PYLOGCONTEXT_H
