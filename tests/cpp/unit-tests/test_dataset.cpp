#include "catch.hpp"
#include "dataset.h"

SCENARIO("Testig DataSet object", "[DataSet]")
{
    GIVEN("A DataSet object")
    {
        std::string dataset_name;
        dataset_name = "simple_dataset";
        SmartRedis::DataSet dataset(dataset_name);
        THEN("The name of the data set can be retrieved")
        {
            CHECK(dataset.name == dataset_name);
        }
        AND_THEN("A tensor can be added to the dataset")
        {
            std::string tensor_name = "tensor_0";
            // ...
        }
    }
}