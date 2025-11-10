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
#include "redis.h"
#include "client.h"
#include "address.h"
#include "logger.h"
#include "logcontext.h"
#include "srobject.h"
#include "configoptions.h"

unsigned long get_time_offset();

using namespace SmartRedis;

// Helper class used for accessing protected members of RedisServer
class TestSSDB : public Redis
{
    public:
        TestSSDB(ConfigOptions* c) : Redis(c) {}

        SRAddress get_ssdb()
        {
            return _get_ssdb();
        }

        void clear_cached_SSDB()
        {
            _cfgopts->_clear_option_from_cache("SSDB");
        }
};

// helper function for putting the SSDB environment
// variable back to its original state
void setenv_ssdb(const char* ssdb)
{
    if (ssdb != nullptr) {
      setenv("SSDB", ssdb, true);
    }
}

SCENARIO("Additional Testing for various SSDBs", "[SSDB]")
{
    std::cout << std::to_string(get_time_offset()) << ": Additional Testing for various SSDBs" << std::endl;
    std::string context("test_ssdb");
    log_data(context, LLDebug, "***Beginning SSDB testing***");
    ConfigOptions* cfgopts = ConfigOptions::create_from_environment("").release();
    LogContext lc("test_ssdb");
    cfgopts->_set_log_context(&lc);

    GIVEN("A TestSSDB object")
    {
        const char* old_ssdb = std::getenv("SSDB");

        INFO("SSDB must be set to a valid host and "\
             "port before running this test.");
        REQUIRE(old_ssdb != NULL);

        TestSSDB test_ssdb(cfgopts);
        Client* c = NULL;

        THEN("SSDB environment variable must exist "
             "and contain valid characters")
        {
            // SSDB is nullptr
            unsetenv("SSDB");
            test_ssdb.clear_cached_SSDB();
            CHECK_THROWS_AS(test_ssdb.get_ssdb(), SmartRedis::RuntimeException);

            // SSDB contains invalid characters
            setenv_ssdb ("127.0.0.1:*&^9");
            test_ssdb.clear_cached_SSDB();
            CHECK_THROWS_AS(test_ssdb.get_ssdb(), SmartRedis::RuntimeException);

            // Valid SSDB. Ensure one of 127 or 128 is chosen
            setenv_ssdb("127,128");
            test_ssdb.clear_cached_SSDB();
            CHECK_THROWS_AS(test_ssdb.get_ssdb(), SmartRedis::RuntimeException);

            // SSDB points to a unix domain socket and we're using clustered Redis
            // FINDME: This test uses a deprecated constructor and will need to be rewritten
            setenv_ssdb ("unix://127.0.0.1:6349");
            CHECK_THROWS_AS(c = new Client(true, "test_ssdb"), SmartRedis::RuntimeException);

            setenv_ssdb(old_ssdb);
        }
    }
    log_data(context, LLDebug, "***End SSDB testing***");
}