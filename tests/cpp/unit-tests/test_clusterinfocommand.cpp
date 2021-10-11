#include "../../../third-party/catch/catch.hpp"
#include "clusterinfocommand.h"

using namespace SmartRedis;

SCENARIO("Parsing an empty string for cluster info")
{
    GIVEN("A ClusterInfoCommand and an empty string")
    {
        ClusterInfoCommand cmd;
        std::string info = "";

        WHEN("calling parse_db_cluster_info on the empty string")
        {
            THEN("An empty parsed_reply_map is returned")
            {
                CHECK(cmd.parse_db_cluster_info(info).size() == 0);
            }
        }
    }
}