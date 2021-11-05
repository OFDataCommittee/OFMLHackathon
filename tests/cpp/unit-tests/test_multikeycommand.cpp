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
#include "multikeycommand.h"


using namespace SmartRedis;

SCENARIO("Adding fields of different types", "[MultiKeyCommand]")
{

    GIVEN("A MultiKeyCommand object")
    {
        MultiKeyCommand cmd;

        WHEN("Fields are added to the MultiKeyCommand object in every possible manner")
        {
            // create the fields
            std::string field_1 = "AI.TENSORSET";
            char field_2[5] = "BLOB";
            const char field_3[4] = "DEL";
            char field_4[13] = "AI.TENSORSET";
            size_t field_size_4 = std::strlen(field_4);
            std::string field_5 = "BLOB";
            std::vector<std::string> fields_1 = {"DEL", "BLOB", "AI.TENSORSET"};

            // define necessary variables for testing
            std::string output = " AI.TENSORSET BLOB DEL AI.TENSORSET BLOB DEL BLOB AI.TENSORSET";
            std::vector<std::string> sorted_keys = {"AI.TENSORSET", "BLOB", "DEL"};
            std::vector<std::string> cmd_keys;

            // add the fields to the Command
            cmd.add_field(field_1, false);
            cmd.add_field(field_2, true);
            cmd.add_field(field_3, true);
            cmd.add_field_ptr(field_4, field_size_4);
            cmd.add_field_ptr(field_5);
            cmd.add_fields(fields_1, true);

            THEN("The MultiKeyCommand object is structured correctly")
            {
                CHECK(cmd.has_keys() == true);
                CHECK(cmd.first_field() == field_1);
                CHECK(output == cmd.to_string());

                cmd_keys = cmd.get_keys();
                std::sort(cmd_keys.begin(), cmd_keys.end());
                CHECK(cmd_keys == sorted_keys);
            }
        }
    }
}