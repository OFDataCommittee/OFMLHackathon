#include "../../../third-party/catch/catch.hpp"
#include "singlekeycommand.h"


using namespace SmartRedis;

SCENARIO("Retrieve field to empty SingleKeyCommand", "[SingleKeyCommand]")
{

    GIVEN("An empty SingleKeyCommand object")
    {
        SingleKeyCommand cmd;

        WHEN("The first field is attempted to be retrieved")
        {

            THEN("A runtime error is thrown")
            {
                CHECK_THROWS_AS(cmd.first_field(), std::runtime_error);
            }
        }
    }

}

SCENARIO("Testing copy constructor for SingleKeyCommand on heap", "[SingleKeyCommand]")
{
    GIVEN("A SingleKeyCommand object on the heap")
    {
        SingleKeyCommand* cmd = new SingleKeyCommand;

        // define fields
        std::string field_1 = "EXISTS";
        char field_2[5] = "BLOB";
        const char field_3[8] = "OUTPUTS";
        char field_4[4] = "TAG";
        size_t field_size_4 = std::strlen(field_4);
        std::string field_5 = "INPUTS";
        std::vector<std::string> fields_1 = {"BATCHSIZE", "BLOB", "OUTPUTS"};
        std::string_view field_sv = std::string_view(field_4, field_size_4);

        // define necessary variables for testing
        std::string output = " EXISTS BLOB OUTPUTS TAG INPUTS "
                             "BATCHSIZE BLOB OUTPUTS TAG";
        std::vector<std::string> sorted_keys =
            {"BATCHSIZE", "BLOB", "OUTPUTS"};
        std::vector<std::string> cmd_keys;

        // add fields
        cmd->add_field(field_1, false);
        cmd->add_field(field_2, true);
        cmd->add_field(field_3, true);
        cmd->add_field_ptr(field_4, field_size_4);
        cmd->add_field_ptr(field_5);
        cmd->add_fields(fields_1, true);
        cmd->add_field_ptr(field_sv);

        THEN("The SingleKeyCommand can be copied with the copy "
                 "constructor and then can be deleted while "
                 "preserving the original SingleKeyCommand's state")
        {
            SingleKeyCommand* cmd_cpy = new SingleKeyCommand(*cmd);

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
            std::sort(cmd_keys.begin(), cmd_keys.end());
            std::sort(cmd_keys_cpy.begin(), cmd_keys_cpy.end());
            CHECK(cmd_keys_cpy == cmd_keys);

            delete cmd;

            // Ensure the state of the original Command object is preserved
            CHECK(cmd_cpy->has_keys() == true);
            CHECK(cmd_cpy->first_field() == field_1);
            CHECK(output == cmd_cpy->to_string());

            cmd_keys = cmd_cpy->get_keys();
            std::sort(cmd_keys.begin(), cmd_keys.end());
            CHECK(cmd_keys == sorted_keys);
            delete cmd_cpy;
        }
    }
}