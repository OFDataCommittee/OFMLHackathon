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

    Uses SmartSim/SmartRedis to train an ML model on a field on a input mesh
    resolution and then approximates the field on the output mesh resolution
    using the ML model.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "wordList.H"
#include "timeSelector.H"
#include "client.h"

// TODO(TM): include smartredis header (see solver and function object example).

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    // Selecting the time step for the input field.
    timeSelector::addOptions();

    argList::addOption
    (
        "inputCase", 
        "inputCase",
        "Path to the input mesh resolution OpenFOAM case." 
    ); 

    argList::addOption
    (
        "outputCase", 
        "outputCase",
        "Path to the output mesh resolution OpenFOAM case." 
    ); 

    argList::addOption
    (
        "field", 
        "field",
        "Name of the mapped field, e.g. p" 
    ); 


    #include "setRootCase.H"
    #include "createTimes.H"
    #include "createMeshes.H"
    #include "createFields.H"

    // TODO(TM): Initialize smartredis client (open channel to smartredis db).  
    
    const volVectorField& inputMeshCellCenters = inputMesh.C();
    const volVectorField& outputMeshCellCenters = outputMesh.C();

    // If 'field' is a volScalarField 
    if (!inputVolScalarFieldTmp->empty())
    {
        // Map the volScalarField using smartsim and smartredis

        //- Put the input mesh cell centers in smartredis.
        
        //- Put the input field in smartredis.
        
        //- (In the Jupyter Notebook - fetch the fields and train an ML model).

        //- Put the output cell centers tensor in smartredis.
        
        //- Forward the trained model on the output cell centers.  

        //- Get the output field from smartredis. 

        //- Read the OpenFOAM output field from the output case.  
        
        //- Assign the smart redis output field values to OpenFOAM output field.

        //- Write the OpenFOAM output field.

        //- (Run the output case and see if the convergence has improved).
        
    }
    // If 'field' is a volVectorField 
    else if (!inputVolVectorFieldTmp->empty())
    {
        // TODO(TM): same as for the volScalarField only with different tensor dimensions.
    }
    else
    {
        // TODO(TM): FoamFatalError - field was not found. 
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    Info<< nl;
    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
