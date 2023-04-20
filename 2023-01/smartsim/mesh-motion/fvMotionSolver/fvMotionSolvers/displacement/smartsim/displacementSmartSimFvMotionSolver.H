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
    Foam::displacementSmartSimFvMotionSolver

Group
    grpMeshMotionSolvers

Description

    Mesh motion solver for an fvMesh. Globally approximates mesh displacement
    using Machine Learning Models. Coupling with ML is done using SmartSim. 

SourceFiles
    displacementSmartSimFvMotionSolver.C

\*---------------------------------------------------------------------------*/

#ifndef displacementSmartSimFvMotionSolver_H
#define displacementSmartSimFvMotionSolver_H

#include "displacementMotionSolver.H"
#include "fvMotionSolver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward class declarations
class motionInterpolation;
class motionDiffusivity;

/*---------------------------------------------------------------------------*\
             Class displacementSmartSimFvMotionSolver Declaration
\*---------------------------------------------------------------------------*/

class displacementSmartSimFvMotionSolver
:
    public displacementMotionSolver,
    public fvMotionSolver
{
    // Private data

        //- Cell-centre motion field
        mutable volVectorField cellDisplacement_;

        //- Optionally read point-position field. Used only for position
        //  boundary conditions.
        mutable autoPtr<pointVectorField> pointLocation_;

        //- Interpolation used to transfer cell displacement to the points
        autoPtr<motionInterpolation> interpolationPtr_;

        //- Diffusivity used to control the motion
        autoPtr<motionDiffusivity> diffusivityPtr_;

        //- Frozen points (that are not on patches). -1 or points that are
        //  fixed to be at points0_ location
        label frozenPointsZone_;


    // Private Member Functions

        //- No copy construct
        displacementSmartSimFvMotionSolver
        (
            const displacementSmartSimFvMotionSolver&
        ) = delete;

        //- No copy assignment
        void operator=(const displacementSmartSimFvMotionSolver&) = delete;


public:

    //- Runtime type information
    TypeName("displacementSmartSim");


    // Constructors

        //- Construct from polyMesh and IOdictionary
        displacementSmartSimFvMotionSolver
        (
            const polyMesh&,
            const IOdictionary&
        );

        //- Construct from components
        displacementSmartSimFvMotionSolver
        (
            const polyMesh& mesh,
            const IOdictionary& dict,
            const pointVectorField& pointDisplacement,
            const pointIOField& points0
        );


    //- Destructor
    ~displacementSmartSimFvMotionSolver();


    // Member Functions

        //- Return reference to the cell motion displacement field
        volVectorField& cellDisplacement()
        {
            return cellDisplacement_;
        }

        //- Return const reference to the cell motion displacement field
        const volVectorField& cellDisplacement() const
        {
            return cellDisplacement_;
        }

        //- Return reference to the diffusivity field
        motionDiffusivity& diffusivity();

        //- Return point location obtained from the current motion field
        virtual tmp<pointField> curPoints() const;

        //- Solve for motion
        virtual void solve();

        //- Update topology
        virtual void updateMesh(const mapPolyMesh&);
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
