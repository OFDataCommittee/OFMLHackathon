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
#include "addressatcommand.h"

using namespace SmartRedis;

SCENARIO("Ensuring the iterators for an AddressAtCommand are correct", "[AddressAtCommand]")
{
    GIVEN("An AddressAtCommand with a single field")
    {
        AddressAtCommand cmd;
        std::string field = "INFO";
        cmd << field;
        WHEN("Iterators from the AddressAtKeyCommand object's "
             "_fields data member are retrieved")
        {
            Command::iterator begin_it = cmd.begin();
            Command::iterator end_it = cmd.end();
            Command::const_iterator cbegin_it = cmd.cbegin();
            Command::const_iterator cend_it = cmd.cend();
            THEN("The iterators that were retrieved are the correct iterators")
            {
                CHECK(*cmd.begin() == field);
                CHECK(*cmd.cbegin() == field);
                begin_it++;
                cbegin_it++;
                CHECK(begin_it == end_it);
                CHECK(cbegin_it == cend_it);
            }
        }
    }
}

SCENARIO("Testing assignment operator for AddressAtCommand on heap", "[AddressAtCommand]")
{
    GIVEN("An AddressAtCommand object on the heap")
    {
        AddressAtCommand* cmd = new AddressAtCommand;

        WHEN("Fields are added to the AddressAtCommand in every possible manner")
        {
            // define fields
            std::string field_1 = "INFO";
            char field_2[8] = "CLUSTER";
            const char field_3[11] = "everything";
            char field_4[5] = "INFO";
            size_t field_size_4 = std::strlen(field_4);
            std::string field_5 = "INFO";
            std::vector<std::string> fields_1 = {"CLUSTER", "everything", "INFO"};
            std::string_view field_sv = std::string_view(field_4, field_size_4);

            // define necessary variables for testing
            std::string output = " INFO CLUSTER everything INFO "
                                "INFO CLUSTER everything INFO INFO";
            std::vector<std::string> keys = {};
            std::vector<std::string> cmd_keys;

            // add fields
            *cmd << field_1 << field_2 << field_3;
            cmd->add_field_ptr(field_4, field_size_4);
            *cmd << field_5 << fields_1 << field_sv;

            THEN("The AddressAtCommand can be copied with the assign "
                "operator and then can be deleted while preserving "
                "the original AddressAtCommand's state")
            {
                AddressAtCommand* cmd_cpy = new AddressAtCommand;
                *cmd_cpy << Keyfield("field_to_be_destroyed");

                *cmd_cpy = *cmd;

                Command::const_iterator it = cmd->cbegin();
                Command::const_iterator it_end = cmd->cend();
                Command::const_iterator it_cpy = cmd_cpy->cbegin();
                Command::const_iterator it_end_cpy = cmd_cpy->cend();

                while (it != it_end) {
                    REQUIRE(it_cpy != it_end_cpy);
                    CHECK(*it == *it_cpy);
                    it++;
                    it_cpy++;
                }
                CHECK(it_cpy == it_end_cpy);

                cmd_keys = cmd->get_keys();
                std::vector<std::string> cmd_keys_cpy = cmd_cpy->get_keys();
                CHECK(cmd_keys_cpy == cmd_keys);

                delete cmd;

                // Ensure the state of the original AddressAtCommand is preserved
                CHECK(cmd_cpy->has_keys() == false);
                CHECK(cmd_cpy->first_field() == field_1);
                CHECK(output == cmd_cpy->to_string());

                cmd_keys = cmd_cpy->get_keys();
                CHECK(cmd_keys == keys);
                delete cmd_cpy;
            }
        }
    }
}

SCENARIO("Testing AddressAtCommand member variables", "[AddressAtCommand]")
{

    GIVEN("An AddressAtCommand object")
    {
        AddressAtCommand* cmd = new AddressAtCommand;

        WHEN("An address and port are not set")
        {

            THEN("The command's address will be an empty string")
            {
                CHECK(cmd->get_address() == "");
            }
        }
    }
}