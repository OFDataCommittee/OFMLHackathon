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
#include "redis.h"
#include "client.h"
#include "address.h"
#include "logger.h"

unsigned long get_time_offset();

using namespace SmartRedis;

SCENARIO("Additional Testing for logging", "[LOG]")
{
    std::cout << std::to_string(get_time_offset()) << ": Additional Testing for logging" << std::endl;
    std::string context("test_logger");
    log_data(context, LLDebug, "***Beginning Logger testing***");

    GIVEN("A Client object")
    {
        Client client(use_cluster(), "test_logger");

        THEN("Logging should be able to be done")
        {
            // log_data()
            log_data(context, LLInfo, "This is data logged at the Info level");

            // log_warning()
            log_warning(context, LLInfo, "This is a warning logged at the Info level");

            // log_error()
            log_error(context, LLInfo, "This is an error logged at the Info level");
        }
    }
    log_data(context, LLDebug, "***End Logger testing***");
}