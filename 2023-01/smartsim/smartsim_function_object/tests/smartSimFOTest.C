#include "IOobject.H"
#include "PstreamReduceOps.H"
#include "catch2/catch_all.hpp"
#include "catch2/catch_test_macros.hpp"
#include "fvCFD.H"
#include "fvMesh.H"

#include <csetjmp>
#include <csignal>
#include <cstdlib>
#include <functional>

using namespace Foam;
extern Time* timePtr;
extern argList* argsPtr;

TEST_CASE("Shared SmartRedis client", "[cavity][serial][parallel]")
{
    dictionary dict;
    dict.set("type", "smartSimFunctionObject");
    dict.set("fieldNames", wordList());
    dict.set("fieldDimensions", labelList());
    functionObjects::smartSimFunctionObject fo1("smartSimFo1", *timePtr, dict);
    functionObjects::smartSimFunctionObject fo2("smartSimFo2", *timePtr, dict);
    REQUIRE(&fo1.redisDB == &fo2.redisDB);
}
