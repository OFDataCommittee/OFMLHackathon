#include "../../../third-party/catch/catch.hpp"
#include "dbinfocommand.h"

using namespace SmartRedis;

SCENARIO("Parsing an empty string for db info")
{
    GIVEN("A DBInfoCommand and an empty string")
    {
        DBInfoCommand cmd;
        std::string info = "";

        WHEN("calling parse_db_node_info on the empty string")
        {
            THEN("An empty info_map is returned")
            {
                CHECK(cmd.parse_db_node_info(info).size() == 0);
            }
        }
    }
}