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

#include "../../../third-party/catch/single_include/catch2/catch.hpp"
#include "../client_test_utils.h"
#include "logger.h"
#include "configoptions.h"

unsigned long get_time_offset();

using namespace SmartRedis;

SCENARIO("Testing for ConfigOptions", "[CfgOpts]")
{
    std::cout << std::to_string(get_time_offset()) << ": Testing for ConfigOptions" << std::endl;
    std::string context("test_configopts");
    log_data(context, LLDebug, "***Beginning ConfigOptions testing***");

    GIVEN("A ConfigOptions object")
    {
        // Make sure keys aren't set before we start
        const char* keys[] = {
            "test_integer_key_that_is_not_really_present",
            "test_string_key_that_is_not_really_present",
            "test_integer_key",
            "test_string_key"
        };
        INFO("Reserved keys must not be set before running this test.");
        for (size_t i = 0; i < sizeof(keys)/sizeof(keys[0]); i++) {
            REQUIRE(std::getenv(keys[i]) == NULL);
        }

        // Set up keys for testing
        setenv("test_integer_key", "42", true);
        setenv("test_string_key", "charizard", true);


        auto co = ConfigOptions::create_from_environment("");

        THEN("Options should be configurable")
        {
            // integer option tests
            CHECK(co->get_integer_option("test_integer_key") == 42);
            CHECK_FALSE(co->is_configured("test_integer_key_that_is_not_really_present"));
            CHECK_THROWS_AS(
                co->get_integer_option(
                    "test_integer_key_that_is_not_really_present"),
                KeyException);
            CHECK(co->_resolve_integer_option(
                "test_integer_key_that_is_not_really_present", 11) == 11);
            CHECK(co->is_configured("test_integer_key_that_is_not_really_present"));
            CHECK(co->get_integer_option(
                "test_integer_key_that_is_not_really_present") == 11);
            co->override_integer_option(
                "test_integer_key_that_is_not_really_present", 42);
            CHECK(co->get_integer_option(
                "test_integer_key_that_is_not_really_present") == 42);
            CHECK(co->_resolve_integer_option(
                "test_integer_key_that_is_not_really_present", 11) == 42);

            // string option tests
            CHECK(co->get_string_option("test_string_key") == "charizard");
            CHECK_FALSE(co->is_configured("test_string_key_that_is_not_really_present"));
            CHECK_THROWS_AS(
                co->get_string_option(
                    "test_string_key_that_is_not_really_present"),
                KeyException);
            CHECK(co->_resolve_string_option(
                "test_string_key_that_is_not_really_present", "pikachu") == "pikachu");
            CHECK(co->is_configured("test_string_key_that_is_not_really_present"));
            CHECK(co->get_string_option(
                "test_string_key_that_is_not_really_present") == "pikachu");
            co->override_string_option(
                "test_string_key_that_is_not_really_present", "meowth");
            CHECK(co->get_string_option(
                "test_string_key_that_is_not_really_present") == "meowth");
            CHECK(co->_resolve_string_option(
                "test_string_key_that_is_not_really_present", "pikachu") == "meowth");
        }
    }

    // Clean up test keys
    unsetenv("test_integer_key");
    unsetenv("test_string_key");

    log_data(context, LLDebug, "***End ConfigOptions testing***");
}

SCENARIO("Suffix Testing for ConfigOptions", "[CfgOpts]")
{
    std::cout << std::to_string(get_time_offset()) << ": Suffix Testing for ConfigOptions" << std::endl;
    std::string context("test_configopts");
    log_data(context, LLDebug, "***Beginning ConfigOptions suffix testing***");

    GIVEN("A ConfigOptions object")
    {
        // Make sure keys aren't set before we start
        const char* keys[] = {
            "integer_key_that_is_not_really_present_suffixtest",
            "string_key_that_is_not_really_present_suffixtest",
            "integer_key_suffixtest",
            "string_key_suffixtest"
        };
        INFO("Reserved keys must not be set before running this test.");
        for (size_t i = 0; i < sizeof(keys)/sizeof(keys[0]); i++) {
            REQUIRE(std::getenv(keys[i]) == NULL);
        }

        // Set up keys for testing
        setenv("integer_key_suffixtest", "42", true);
        setenv("string_key_suffixtest", "charizard", true);

        auto co = ConfigOptions::create_from_environment("suffixtest");

        THEN("Suffixed options should be configurable")
        {
            // integer option tests
            CHECK(co->get_integer_option("integer_key") == 42);
            CHECK_FALSE(co->is_configured("integer_key_that_is_not_really_present"));
            CHECK_THROWS_AS(
                co->get_integer_option(
                    "integer_key_that_is_not_really_present"),
                KeyException);
            CHECK(co->_resolve_integer_option(
                "integer_key_that_is_not_really_present", 11) == 11);
            CHECK(co->is_configured("integer_key_that_is_not_really_present"));
            CHECK(co->get_integer_option(
                "integer_key_that_is_not_really_present") == 11);
            co->override_integer_option(
                "integer_key_that_is_not_really_present", 42);
            CHECK(co->get_integer_option(
                "integer_key_that_is_not_really_present") == 42);
            CHECK(co->_resolve_integer_option(
                "integer_key_that_is_not_really_present", 11) == 42);

            // string option tests
            CHECK(co->get_string_option("string_key") == "charizard");
            CHECK_FALSE(co->is_configured("string_key_that_is_not_really_present"));
            CHECK_THROWS_AS(
                co->get_string_option(
                    "string_key_that_is_not_really_present"),
                KeyException);
            CHECK(co->_resolve_string_option(
                "string_key_that_is_not_really_present", "pikachu") == "pikachu");
            CHECK(co->is_configured("string_key_that_is_not_really_present"));
            CHECK(co->get_string_option(
                "string_key_that_is_not_really_present") == "pikachu");
            co->override_string_option(
                "string_key_that_is_not_really_present", "meowth");
            CHECK(co->get_string_option(
                "string_key_that_is_not_really_present") == "meowth");
            CHECK(co->_resolve_string_option(
                "string_key_that_is_not_really_present", "pikachu") == "meowth");
        }
    }

    // Clean up test keys
    unsetenv("integer_key_suffixtest");
    unsetenv("string_key_suffixtest");

    log_data(context, LLDebug, "***End ConfigOptions suffix testing***");
}

