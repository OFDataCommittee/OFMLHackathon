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
void putenv_ssdb(const char* ssdb)
{
    if (ssdb != nullptr) {
        std::string reset_ssdb = std::string("SSDB=") + std::string(ssdb);
        char* reset_ssdb_c = new char[reset_ssdb.size() + 1];
        std::copy(reset_ssdb.begin(), reset_ssdb.end(), reset_ssdb_c);
        reset_ssdb_c[reset_ssdb.size()] = '\0';
        putenv(reset_ssdb_c);
        delete [] reset_ssdb_c;
    }
}

SCENARIO("Additional Testing for various SSDBs", "[SSDB]")
{

    GIVEN("A TestSSDB object")
    {
        const char* old_ssdb = std::getenv("SSDB");
        // SSDB can't be nullptr or invalid upon instantiation of TestSSDB
        // object, so temporarily set SSDB to a valid string
        putenv_ssdb("127.0.0.1:6379");

        TestSSDB test_ssdb;

        THEN("SSDB environment variable must exist "
             "and contain valid characters")
        {
            // SSDB is nullptr
            unsetenv("SSDB");
            CHECK_THROWS(test_ssdb.get_ssdb());

            // SSDB contains invalid characters
            putenv_ssdb("127.0.0.1:*&^9");
            CHECK_THROWS(test_ssdb.get_ssdb());

            // Valid SSDB. Ensure one of 127 or 128 is chosen
            putenv_ssdb("127,128");
            std::string hp = test_ssdb.get_ssdb();
            CHECK((hp == "tcp://127" || hp == "tcp://128"));

            putenv_ssdb(old_ssdb);
        }
    }
}
