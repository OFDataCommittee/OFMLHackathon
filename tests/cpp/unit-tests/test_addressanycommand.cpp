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

#include "../../../third-party/catch/single_include/catch2/catch.hpp"
#include "addressanycommand.h"
#include "redisserver.h"

using namespace SmartRedis;

SCENARIO("Testing assignment operator for AddressAnyCommand", "[AddressAnyCommand]")
{

    GIVEN("An AddressAnyCommand object")
    {
        AddressAnyCommand cmd;

        WHEN("Fields are added to the AddressAnyCommand in every possible manner")
        {
            // create the fields
            std::string field_1 = "FLUSHALL";
            char field_2[5] = "INFO";
            const char field_3[5] = "TEST";
            char field_4[9] = "FLUSHALL";
            size_t field_size_4 = std::strlen(field_4);
            std::string field_5 = "INFO";
            std::vector<std::string> fields_1 = {"TEST", "FLUSHALL", "INFO"};

            // define necessary variables for testing
            std::string output = " FLUSHALL INFO TEST FLUSHALL INFO TEST FLUSHALL INFO";
            std::vector<std::string> sorted_keys = {};
            std::vector<std::string> cmd_keys;

            // add the fields to the Command
            cmd.add_field(field_1, false);
            cmd.add_field(field_2);
            cmd.add_field(field_3);
            cmd.add_field_ptr(field_4, field_size_4);
            cmd.add_field_ptr(field_5);
            cmd.add_fields(fields_1);


            THEN("The AddressAnyCommand object can be copied "
                     "with the assignment operator")
            {
                AddressAnyCommand* cmd_cpy = new AddressAnyCommand;
                cmd_cpy->add_field("field_to_be_destroyed", true);

                *cmd_cpy = cmd;

                Command::const_iterator it = cmd.cbegin();
                Command::const_iterator it_end = cmd.cend();
                Command::const_iterator it_cpy = cmd_cpy->cbegin();
                Command::const_iterator it_end_cpy = cmd_cpy->cend();

                while (it != it_end) {
                    REQUIRE(it_cpy != it_end_cpy);
                    CHECK(*it == *it_cpy);
                    it++;
                    it_cpy++;
                }
                CHECK(it_cpy == it_end_cpy);

                cmd_keys = cmd.get_keys();
                std::vector<std::string> cmd_keys_cpy = cmd_cpy->get_keys();
                CHECK(cmd_keys_cpy == cmd_keys);

                delete cmd_cpy;

                // Ensure the state of the original Command object is preserved
                CHECK(cmd.has_keys() == false);
                CHECK(cmd.first_field() == field_1);
                CHECK(output == cmd.to_string());

                cmd_keys = cmd.get_keys();
                CHECK(cmd_keys == sorted_keys);
            }
        }
    }
}

SCENARIO("Testing AddressAnyCommand member variables", "[AddressAnyCommand]")
{

    GIVEN("An AddressAnyCommand object and a db node address and port")
    {
        AddressAnyCommand cmd;
        std::string db_address = "127.0.0.1";
        uint16_t db_port = 6379;

        WHEN("An address and port are set")
        {
            cmd.set_exec_address_port(db_address, db_port);

            THEN("The command's address and port will be the same as those set")
            {
                CHECK(cmd.get_address() == db_address);
                CHECK(cmd.get_port() == db_port);
            }
        }
    }
}