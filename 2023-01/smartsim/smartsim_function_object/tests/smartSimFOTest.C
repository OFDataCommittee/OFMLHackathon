#include "IOobject.H"
#include "PstreamReduceOps.H"
#include "catch2/catch_all.hpp"
#include "catch2/catch_test_macros.hpp"
#include "fvCFD.H"
#include "fvMesh.H"

#include "smartSimFunctionObject.H"
#include "functionObjectList.H"

using namespace Foam;
extern Time* timePtr;
extern argList* argsPtr;

TEST_CASE("Shared SmartRedis client", "[cavity][serial][parallel]")
{
    Time& runTime = *timePtr;
    FatalError.dontThrowExceptions();
    fvMesh mesh
    (
        IOobject
        (
            polyMesh::defaultRegion,
            runTime.constant(),
            runTime,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    dictionary dict0;
    dict0.set("region", polyMesh::defaultRegion);
    dict0.set("type", "smartSimFunctionObject");
    dict0.set("fieldNames", wordList());
    dict0.set("fieldDimensions", labelList());
    dict0.set("clusterMode", false);
    dict0.set("clientName", "default");
    dictionary dict1;
    dict1.set("region", polyMesh::defaultRegion);
    dict1.set("type", "smartSimFunctionObject");
    dict1.set("fieldNames", wordList());
    dict1.set("fieldDimensions", labelList());
    dict1.set("clusterMode", false);
    dict1.set("clientName", "default");
    functionObjects::smartSimFunctionObject o0("smartSim0", runTime, dict0);
    functionObjects::smartSimFunctionObject o1("smartSim1", runTime, dict1);
    REQUIRE(&o0.client() == &o1.client());
}
