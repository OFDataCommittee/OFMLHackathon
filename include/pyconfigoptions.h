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

#ifndef SMARTREDIS_PYCONFIGOPTIONS_H
#define SMARTREDIS_PYCONFIGOPTIONS_H


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include "configoptions.h"
#include "pysrobject.h"

///@file

namespace py = pybind11;


namespace SmartRedis {


/*!
*   \brief The PyConfigOptions class is a wrapper around the
           C++ ConfigOptions class.
*/
class PyConfigOptions
{
    public:

        /*!
        *   \brief PyConfigOptions constructor
        */
        PyConfigOptions();

        /*!
        *   \brief PyConfigOptions constructor from a
        *          SmartRedis::ConfigOptions object
        *   \param configoptions A SmartRedis::ConfigOptions object
        *                        allocated on the heap.  The SmartRedis
        *                        PConfigOptions will be deleted upon
        *                        PyConfigOptions deletion.
        */
        PyConfigOptions(ConfigOptions* configoptions);

        /*!
        *   \brief PyConfigOptions destructor
        */
        virtual ~PyConfigOptions();

        /*!
        *   \brief Retrieve a pointer to the underlying
        *          SmartRedis::ConfigOptions object
        *   \returns ConfigOptions pointer within PyConfigOptions
        */
        ConfigOptions* get();

        /////////////////////////////////////////////////////////////
        // Factory construction methods

        /*!
        *   \brief Instantiate ConfigOptions, getting selections from
        *          environment variables. If \p db_suffix is non-empty,
        *          then "_{db_suffix}" will be appended to the name of
        *          each environment variable that is read.
        *   \param db_suffix The suffix to use with environment variables,
        *                    or an empty string to disable suffixing
        *   \returns The constructed ConfigOptions object
        *   \throw SmartRedis::Exception if db_suffix contains invalid
        *          characters
        */
        static PyConfigOptions* create_from_environment(
            const std::string& db_suffix);

        /////////////////////////////////////////////////////////////
        // Option access

        /*!
        *   \brief Retrieve the value of a numeric configuration option
        *          from the selected source
        *   \param option_name The name of the configuration option to retrieve
        *   \returns The value of the selected option. Returns
        *            \p default_value if the option was not set in the
        *            selected source
        */
        int64_t get_integer_option(const std::string& option_name);

        /*!
        *   \brief Retrieve the value of a string configuration option
        *          from the selected source
        *   \param option_name The name of the configuration option to retrieve
        *   \returns The value of the selected option. Returns
        *            \p default_value if the option was not set in the
        *            selected source
        */
        std::string get_string_option(const std::string& option_name);

        /*!
        *   \brief Check whether a configuration option is set in the
        *          selected source
        *   \param option_name The name of the configuration option to check
        *   \returns True IFF the target option is defined in the selected
        *            source
        */
        bool is_configured(const std::string& option_name);

        /////////////////////////////////////////////////////////////
        // Option overrides

        /*!
        *   \brief Override the value of a numeric configuration option
        *          in the selected source
        *   \details Overrides are specific to an instance of the
        *            ConfigOptions class. An instance that references
        *            the same source will not be affected by an override to
        *            a different ConfigOptions instance
        *   \param option_name The name of the configuration option to override
        *   \param value The value to store for the configuration option
        */
        void override_integer_option(
            const std::string& option_name, int64_t value);

        /*!
        *   \brief Override the value of a string configuration option
        *          in the selected source
        *   \details Overrides are specific to an instance of the
        *            ConfigOptions class. An instance that references
        *            the same source will not be affected by an override to
        *            a different ConfigOptions instance
        *   \param option_name The name of the configuration option to override
        *   \param value The value to store for the configuration option
        */
        void override_string_option(
            const std::string& option_name, const std::string& value);

    private:

        ConfigOptions* _configoptions;
};

} // namespace SmartRedis

#endif // SMARTREDIS_PYCONFIGOPTIONS_H
