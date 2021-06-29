#include "catch.hpp"
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
            CHECK( node_1 < node_2);
        }
    }
    AND_GIVEN("Two DBNode objects created with connection and hash slot information")
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