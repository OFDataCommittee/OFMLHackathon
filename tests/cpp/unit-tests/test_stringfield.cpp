#include "../../../third-party/catch/catch.hpp"
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