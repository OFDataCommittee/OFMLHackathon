#include "catch.hpp"
#include "../client_test_utils.h"
#include "client.h"
#include "dataset.h"
// #include "client_test_utils.h"
// #include "dataset_test_utils.h"

SCENARIO("Testing Client Object", "[Client]")
{
    GIVEN("A Client object not connected to a redis cluster")
    {
        bool use_cluster = false;
        SmartRedis::Client client(use_cluster);
        WHEN("A dataset is put")
        {
            // ...
        }
    }
}