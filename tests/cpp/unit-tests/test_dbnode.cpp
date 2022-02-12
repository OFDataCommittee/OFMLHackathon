/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2022, Hewlett Packard Enterprise
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
#include "dbnode.h"

using namespace SmartRedis;

SCENARIO("Testing DBNode object", "[DBNode]")
{

    GIVEN("Two DBNode objects created with the default contructor")
    {
        DBNode node_1;
        DBNode node_2;

        THEN("The DBNode's data members are set to their default values")
        {
            CHECK(node_1.name == "");
            CHECK(node_1.ip == "");
            CHECK(node_1.port == -1);
            CHECK(node_1.lower_hash_slot == -1);
            CHECK(node_1.upper_hash_slot == -1);
        }

        AND_THEN("The < operator is overloaded correctly")
        {
            CHECK_FALSE(node_1 < node_2);
            node_1.lower_hash_slot = 1;
            node_2.lower_hash_slot = 2;
            CHECK(node_1 < node_2);
        }
    }

    AND_GIVEN("Two DBNode objects created with connection "
              "and hash slot information")
    {
        std::string name = "name";
        std::string ip = "192.168.4.1";
        uint64_t port = -1;
        uint64_t l_slot_1 = 2;
        uint64_t l_slot_2 = 1;
        uint64_t u_slot = -1;
        std::string prefix = "prefix";
        DBNode node_1(ip, name, port, l_slot_1, u_slot, prefix);
        DBNode node_2(ip, name, port, l_slot_2, u_slot, prefix);

        THEN("The DBNode's data members are set to their default values")
        {
            CHECK(node_1.name == name);
            CHECK(node_1.ip == ip);
            CHECK(node_1.port == port);
            CHECK(node_1.lower_hash_slot == l_slot_1);
            CHECK(node_1.upper_hash_slot == u_slot);
            CHECK(node_1.prefix == prefix);
        }

        AND_THEN("The < operator is overloaded correctly")
        {
            CHECK_FALSE(node_1 < node_2);
            node_1.lower_hash_slot = 1;
            node_2.lower_hash_slot = 2;
            CHECK(node_1 < node_2);
        }
    }
}