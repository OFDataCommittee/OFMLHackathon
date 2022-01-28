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
#include "stringfield.h"

SCENARIO("Test StringField", "[StringField]")
{

    GIVEN("A StringField object constructed with the string field name")
    {
        std::string name_1 = "stringfield_name_1";
        StringField stringfield_1(name_1);

        THEN("The object is created correctly")
        {
            CHECK(stringfield_1.name() == name_1);
            CHECK(stringfield_1.size() == 0);
        }

        AND_THEN("The StringField object can be manipulated")
        {
            std::string val = "100";
            std::vector<std::string> vals{val};
            const std::vector<std::string> c_vals{val};

            stringfield_1.append(val);
            CHECK(stringfield_1.size() == 1);
            CHECK(vals.size() == 1);
            CHECK(stringfield_1.values() == vals);
            CHECK(stringfield_1.immutable_values() == c_vals);

            stringfield_1.clear();
            CHECK(stringfield_1.size() == 0);
            CHECK(stringfield_1.values().size() == 0);
            CHECK(stringfield_1.immutable_values().size() == 0);
        }
    }

    GIVEN("A StringField object constructed with the"
          "string field name and values to be copied")
    {
        std::string name_2 = "stringfield_name_2";
        std::vector<std::string> vals_to_cpy{"100", "200", "300"};
        StringField stringfield_2(name_2, vals_to_cpy);

        THEN("The object is created correctly")
        {
            CHECK(stringfield_2.name() == name_2);
            CHECK(stringfield_2.size() == 3);
            CHECK(stringfield_2.values() == vals_to_cpy);
        }
    }

    GIVEN("A StringField object constructed with the"
          "string field name and values to be moved")
    {
        std::string name_3 = "stringfield_name_3";
        std::vector<std::string> vals_to_move{"100", "200", "300"};
        StringField stringfield_3(name_3, std::move(vals_to_move));

        THEN("The object is created correctly")
        {
            std::vector<std::string> expected_vals{"100", "200", "300"};
            CHECK(stringfield_3.name() == name_3);
            CHECK(stringfield_3.size() == 3);
            CHECK(stringfield_3.values() == expected_vals);
        }
        // TODO: Test serializing the StringField
    }
}