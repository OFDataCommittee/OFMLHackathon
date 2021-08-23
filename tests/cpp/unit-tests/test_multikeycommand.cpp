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
            std::vector<std::string> sorted_keys =
                {"AI.TENSORSET", "BLOB", "DEL"};
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