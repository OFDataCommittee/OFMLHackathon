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

#include "../../../third-party/catch/catch.hpp"
#include "redis.h"

using namespace SmartRedis;

// Helper class used for accessing protected members of RedisServer
class TestSSDB : public Redis
{
    public:
        TestSSDB() : Redis() {}

        std::string get_ssdb()
        {
            return this->_get_ssdb();
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

    GIVEN("A TestSSDB object")
    {
        const char* old_ssdb = std::getenv("SSDB");

    INFO("SSDB must be set to a valid host and "\
         "port before running this test.");
    REQUIRE(old_ssdb != NULL);

        TestSSDB test_ssdb;

        THEN("SSDB environment variable must exist "
             "and contain valid characters")
        {
            // SSDB is nullptr
            unsetenv("SSDB");
            CHECK_THROWS(test_ssdb.get_ssdb());

            // SSDB contains invalid characters
            setenv_ssdb ("127.0.0.1:*&^9");
            CHECK_THROWS(test_ssdb.get_ssdb());

            // Valid SSDB. Ensure one of 127 or 128 is chosen
            setenv_ssdb("127,128");
            std::string hp = test_ssdb.get_ssdb();
            CHECK((hp == "tcp://127" || hp == "tcp://128"));

            setenv_ssdb(old_ssdb);
        }
    }
}