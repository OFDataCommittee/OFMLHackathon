/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2023 TU Darmstadt  
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
    foamSmartSimMapFields

Description

    Uses SmartSim/SmartRedis to train an ML model on a field on a coarse mesh
    resolution and then approximates the field on the fine mesh resolution
    using the ML model.

\*---------------------------------------------------------------------------*/


Info<< "Creating input mesh\n" << endl;

fvMesh inputMesh 
(
    IOobject
    (
        fvMesh::defaultRegion,
        runTimeInput.timeName(),
        runTimeInput, 
        IOobject::MUST_READ, 
        IOobject::AUTO_WRITE
    )
);

Info<< "Creating output mesh\n" << endl;
fvMesh outputMesh
(
    IOobject
    (
        fvMesh::defaultRegion,
        runTimeOutput.timeName(),
        runTimeOutput, 
        IOobject::MUST_READ, 
        IOobject::AUTO_WRITE
    )
);

// ************************************************************************* //
