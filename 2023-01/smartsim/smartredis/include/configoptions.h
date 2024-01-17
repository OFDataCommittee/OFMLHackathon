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

#ifndef SMARTREDIS_CONFIGOPTIONS_H
#define SMARTREDIS_CONFIGOPTIONS_H

#ifndef __cplusplus
#error C users should include c_configoptions.h, not configoptions.h
#endif

#ifdef __cplusplus
#include <string>
#include <memory> // for unique_ptr
#include <unordered_map>
#include "srobject.h"
#include "sr_enums.h"
#include "srexception.h"

///@file

namespace SmartRedis {

/*!
*   \brief Configuration source enumeration
*/
enum cfgSrc {
    cs_envt,    // Configuration data is coming from environment variables
};


/*!
*   \brief The ConfigOptions class consolidates access to configuration options
*          used in SmartRedis
*/
class ConfigOptions
{
    private:
        /*!
        *   \brief ConfigOptions constructor. Do not use! To instantiate a
        *          ConfigOptions object, use one of the factory methods below
        *   \param source The selected source for config data
        *   \param string The string associated with the source
        */
        ConfigOptions(cfgSrc source, const std::string& string);

    public:

        /*!
        *   \brief ConfigOptions copy constructor
        *   \param cfgopts The ConfigOptions to copy
        */
        ConfigOptions(const ConfigOptions& cfgopts) = default;

        /*!
        *   \brief ConfigOptions copy assignment operator
        *   \param cfgopts The ConfigOptions to copy and assign
        */
        ConfigOptions& operator=(const ConfigOptions& cfgopts) = default;

        /*!
        *   \brief ConfigOptions move constructor
        *   \param cfgopts The ConfigOptions to move
        */
        ConfigOptions(ConfigOptions&& cfgopts) = default;

        /*!
        *   \brief ConfigOptions move assignment operator
        *   \param cfgopts The ConfigOptions to move and assign
        */
        ConfigOptions& operator=(ConfigOptions&& cfgopts) = default;

        /*!
        *   \brief ConfigOptions destructor
        */
        virtual ~ConfigOptions();

        /*!
        *   \brief Deep copy a ConfigOptions object
        *   \returns The cloned object
        *   \throw std::bad_alloc on allocation failure
        */
       ConfigOptions* clone();

        /////////////////////////////////////////////////////////////
        // Factory construction methods

        /*!
        *   \brief Instantiate ConfigOptions, getting selections from
        *          environment variables. If \p db_suffix is non-empty,
        *          then "{db_suffix}_" will be prepended to the name of
        *          each environment variable that is read.
        *   \param db_suffix The suffix to use with environment variables,
        *                    or an empty string to disable suffixing
        *   \returns The constructed ConfigOptions object
        *   \throw SmartRedis::Exception if db_suffix contains invalid
        *          characters
        */
        static std::unique_ptr<ConfigOptions> create_from_environment(
            const std::string& db_suffix);

        /*!
        *   \brief Instantiate ConfigOptions, getting selections from
        *          environment variables. If \p db_suffix is non-empty,
        *          then "{db_suffix}_" will be prepended to the name of
        *          each environment variable that is read.
        *   \param db_suffix The suffix to use with environment variables,
        *                    or an empty string to disable suffixing
        *   \returns The constructed ConfigOptions object
        *   \throw SmartRedis::Exception if db_suffix contains invalid
        *          characters
        */
        static std::unique_ptr<ConfigOptions> create_from_environment(
            const char* db_suffix);

        /////////////////////////////////////////////////////////////
        // Option access

        /*!
        *   \brief Retrieve the value of a numeric configuration option
        *          from the selected source
        *   \param option_name The name of the configuration option to retrieve
        *   \returns The value of the selected option
        *   \throw Throws SRKeyException if the option was not set in the
        *          selected source
        */
        int64_t get_integer_option(const std::string& option_name);

        /*!
        *   \brief Retrieve the value of a string configuration option
        *          from the selected source
        *   \param option_name The name of the configuration option to retrieve
        *   \returns The value of the selected option
        *   \throw Throws SRKeyException if the option was not set in the
        *          selected source
        */
        std::string get_string_option(const std::string& option_name);

        /*!
        *   \brief Resolve the value of a string configuration option
        *          from the selected source, selecting a default value
        *          if not configured
        *   \param option_name The name of the configuration option to retrieve
        *   \param default_value The baseline value of the configuration
        *          option to be returned if a value was not set in the
        *          selected source
        *   \returns The value of the selected option. Returns
        *            \p default_value if the option was not set in the
        *            selected source
        */
        std::string _resolve_string_option(
            const std::string& option_name, const std::string& default_value);

        /*!
        *   \brief Resolve the value of a numeric configuration option
        *          from the selected source, selecting a default value
        *          if not configured
        *   \param option_name The name of the configuration option to retrieve
        *   \param default_value The baseline value of the configuration
        *          option to be returned if a value was not set in the
        *          selected source
        *   \returns The value of the selected option. Returns
        *            \p default_value if the option was not set in the
        *            selected source
        */
        int64_t _resolve_integer_option(
            const std::string& option_name, int64_t default_value);

        /*!
        *   \brief Check whether a configuration option is set in the
        *          selected source
        *   \param option_name The name of the configuration option to check
        *   \returns True IFF the target option is defined in the selected
        *            source
        */
        bool is_configured(const std::string& option_name);

        /*!
        *   \brief Retrieve the logging context
        *   \returns The log context associated with this object
        */
        SRObject* _get_log_context() {
            if (_log_context == NULL) {
                throw SRRuntimeException(
                    "Attempt to _get_log_context() before context was set!");
            }
            return _log_context;
        }

        /*!
        *   \brief Store the logging context
        *   \param log_context The context to associate with logging
        */
        void _set_log_context(SRObject* log_context) {
            _log_context = log_context;
        }

        /*!
        *   \brief Clear a configuration option from the cache
        *   \param option_name The name of the option to clear
        */
        void _clear_option_from_cache(const std::string& option_name);

        /*!
        *   \brief Stash a string buffer so we can delete it on cleanup
        *   \param buf The buffer to store
        */
        void _add_string_buffer(char* buf) {
            _string_buffer_stash.push_back(buf);
        }

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

        /*!
        *   \brief Process option data from a fixed source
        *   \throw SmartRedis::Exception: sources other than environment
        *          variables are not implemented yet
        */
        void _populate_options();

        /*!
        *   \brief Apply a suffix to an option name if the source is environment
        *          variables and the suffix is nonempty
        *   \param option_name The name of the option to suffix
        */
        std::string _suffixed(const std::string& option_name);

        /*!
        *  \brief Integer option map
        */
        std::unordered_map<std::string, int64_t> _int_options;

        /*!
        *  \brief String option map
        */
        std::unordered_map<std::string, std::string> _string_options;

        /*!
        *  \brief Configuration source
        */
        cfgSrc _source;

        /*!
        *  \brief Configuration string. Meaning is specific to the source
        */
       std::string _string;

        /*!
        *  \brief Lazy evaluation (do we read in all options at once or only
        *         on demand)
        */
        bool _lazy;

        /*!
        *  \brief Logging context
        */
        SRObject* _log_context;

        /*!
        *  \brief Stash of string buffers to free at cleanup time
        */
       std::vector<char*> _string_buffer_stash;
};

} // namespace SmartRedis

#endif
#endif // SMARTREDIS_CONFIGOPTIONS_H
