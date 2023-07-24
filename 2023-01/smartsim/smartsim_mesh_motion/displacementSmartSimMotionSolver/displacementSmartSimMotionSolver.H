/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2023 Tomislav Maric, TU Darmstadt 
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

Class
    Foam::displacementSmartSimMotionSolver

Group
    grpMeshMotionSolvers

Description

    Mesh motion solver for an fvMesh. Globally approximates mesh displacement
    using Machine Learning Models. Coupling with ML is done using SmartSim. 

SourceFiles
    displacementSmartSimMotionSolver.C

\*---------------------------------------------------------------------------*/

#ifndef displacementSmartSimMotionSolver_H
#define displacementSmartSimMotionSolver_H

#include "displacementMotionSolver.H"
#include "client.h"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward class declarations
class motionInterpolation;
class motionDiffusivity;

/*---------------------------------------------------------------------------*\
             Class displacementSmartSimMotionSolver Declaration
\*---------------------------------------------------------------------------*/

class displacementSmartSimMotionSolver
:
    public displacementMotionSolver
{

    // Private Member Functions
    
        //- Set to false if not using a clustered database 
        bool clusterMode_;
        
        //- SmartRedis Database Client
        SmartRedis::Client client_;

        //- No copy construct
        displacementSmartSimMotionSolver
        (
            const displacementSmartSimMotionSolver&
        ) = delete;

        //- No copy assignment
        void operator=(const displacementSmartSimMotionSolver&) = delete;

public:

    //- Runtime type information
    TypeName("displacementSmartSim");


    // Constructors

        //- Construct from polyMesh and IOdictionary
        displacementSmartSimMotionSolver
        (
            const polyMesh&,
            const IOdictionary&
        );

        //- Construct from components
        displacementSmartSimMotionSolver
        (
            const polyMesh& mesh,
            const IOdictionary& dict,
            const pointVectorField& pointDisplacement,
            const pointIOField& points0
        );


    //- Destructor
    ~displacementSmartSimMotionSolver();

    // Member Functions

    //- Return point location obtained from the current motion field
    virtual tmp<pointField> curPoints() const;

    //- Solve for motion
    virtual void solve(); 
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
