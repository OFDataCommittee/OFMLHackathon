#include "../../../third-party/catch/catch.hpp"
#include "redis.h"

using namespace SmartRedis;

// Helper class used for accessing protected members of RedisServer
class TestSSDB : public Redis
{
    public:
        TestSSDB() : Redis() {}

        std::string get_ssdb()
        {
            return this->_get_ssdb();
        }
};

// helper function for putting the SSDB environment
// variable back to its original state
void setenv_ssdb(const char* ssdb)
{
    if (ssdb != nullptr) {
      setenv("SSDB", ssdb, true);
    }
}

SCENARIO("Additional Testing for various SSDBs", "[SSDB]")
{

    GIVEN("A TestSSDB object")
    {
        const char* old_ssdb = std::getenv("SSDB");

	INFO("SSDB must be set to a valid host and "\
	     "port before running this test.");
	REQUIRE(old_ssdb != NULL);
	  
        TestSSDB test_ssdb;

        THEN("SSDB environment variable must exist "
             "and contain valid characters")
        {
            // SSDB is nullptr
            unsetenv("SSDB");
            CHECK_THROWS(test_ssdb.get_ssdb());

            // SSDB contains invalid characters
            setenv_ssdb ("127.0.0.1:*&^9");
            CHECK_THROWS(test_ssdb.get_ssdb());

            // Valid SSDB. Ensure one of 127 or 128 is chosen
            setenv_ssdb("127,128");
            std::string hp = test_ssdb.get_ssdb();
            CHECK((hp == "tcp://127" || hp == "tcp://128"));

            setenv_ssdb(old_ssdb);
        }
    }
}
