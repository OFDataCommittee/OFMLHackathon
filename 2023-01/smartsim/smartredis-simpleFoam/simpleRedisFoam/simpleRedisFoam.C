/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    simpleRedisFoam

Description
    simpleFoam with a connection to a Redis DB

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "fvOptions.H"

// SmartRedis includes
#include "client.h"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Steady-state solver for incompressible, turbulent flows."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "initContinuityErrs.H"

    turbulence->validate();

    // Create Redis client
    Info<< "Creating SmartRedis client..." << endl;
    SmartRedis::Client client(false);

    // Dimensions of communicated fields
    std::vector<size_t> dims = {1, 1, 1};
    dims[0] = mesh.nCells();

    Info<< "\nStarting time loop\n" << endl;

    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity SIMPLE corrector
        {
            #include "UEqn.H"
            #include "pEqn.H"
        }

        laminarTransport.correct();
        turbulence->correct();

        // Store fields in the DB
        if (Pstream::master()) {
        }

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    // Add p to the database
    client.put_tensor(p.name(), (void*)p.internalField().cdata(), dims,
                      SRTensorTypeDouble, SRMemLayoutContiguous);

    client.put_tensor("x", (void*)mesh.C().component(0)().cdata(), dims,
                      SRTensorTypeDouble, SRMemLayoutContiguous);
    client.put_tensor("y", (void*)mesh.C().component(1)().cdata(), dims,
                      SRTensorTypeDouble, SRMemLayoutContiguous);


    // Fetch p from database/dataset
    std::vector<scalar> unpack_p(mesh.nCells(), -100.0);
    client.unpack_tensor(p.name(),
                            unpack_p.data(),
                            {dims[0]},
                            SRTensorTypeDouble,
                            SRMemLayoutContiguous);

    // Check if fetched values reflect the real ones
    // MAY BE verlunable to floating point issue, but who cares  
    bool isSame = true;
    forAll(p.internalField(), ci) {
        if (p.internalField()[ci] != unpack_p[ci]) {
            isSame = false;
            break;
        }
    }

    Info<< "Check for consistent p values: " << isSame << nl << endl;

    Info<< "End\n" << endl;
    
    return 0;
}


// ************************************************************************* //
