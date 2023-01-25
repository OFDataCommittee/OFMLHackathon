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

#ifndef SMARTREDIS_PYSROBJECT_H
#define SMARTREDIS_PYSROBJECT_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "srobject.h"

///@file

namespace py = pybind11;

namespace SmartRedis {

/*!
*   \brief The PySRObject class is a wrapper around the
           C++ SRObject class.
*/
class PySRObject
{
    public:

        /*!
        *   \brief PySRObject constructor
        *   \param context The context to use for log messages
        */
        PySRObject(const std::string& context);

        /*!
        *   \brief PySRObject constructor from a
        *          SmartRedis::SRObject object
        *   \param logcontext A SmartRedis::SRObject pointer to
        *                  a SmartRedis::SRObject allocated on
        *                  the heap.  The SmartRedis::SRObject
        *                  will be deleted upton PySRObject
        *                  deletion.
        */
        PySRObject(SRObject* srobject);

        /*!
        *   \brief PySRObject destructor
        */
        ~PySRObject();

        /*!
        *   \brief Retrieve a pointer to the underlying
        *          SmartRedis::SRObject object
        *   \returns SRObject pointer within PySRObject
        */
        SRObject* get();

        /*!
        *   \brief Conditionally log data if the logging level is high enough
        *   \param level Minimum logging level for data to be logged
        *   \param data Text of data to be logged
        */
        virtual void log_data(
            SRLoggingLevel level, const std::string& data) const;

        /*!
        *   \brief Conditionally log warning data if the logging level is
        *          high enough
        *   \param level Minimum logging level for data to be logged
        *   \param data Text of data to be logged
        */
        virtual void log_warning(
            SRLoggingLevel level, const std::string& data) const;

        /*!
        *   \brief Conditionally log error data if the logging level is
        *          high enough
        *   \param level Minimum logging level for data to be logged
        *   \param data Text of data to be logged
        */
        virtual void log_error(
            SRLoggingLevel level, const std::string& data) const;

    private:

        SRObject* _srobject;
};

} // namespace SmartRedis

#endif // SMARTREDIS_PYSROBJECT_H
