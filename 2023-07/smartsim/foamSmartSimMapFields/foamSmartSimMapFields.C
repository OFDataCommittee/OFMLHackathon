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

#include "error.H"
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

    const word field = args.get<word>("field");

    #include "createFields.H"

    SmartRedis::Client smartRedisClient(false);

    // If 'field' is a volScalarField 
    if (!inputVolScalarFieldTmp->empty())
    {
        // Let python know the rank of the training data tensor (scalar, vector).
        std::vector<int> tensor_rank {0};
        smartRedisClient.put_tensor("input_field_rank",
                                    (void*)tensor_rank.data(), 
                                    std::vector<size_t>{1},
                                    SRTensorTypeInt32, SRMemLayoutContiguous);
                                    
        // Create the cell centers DataSet
        const auto mpiIndexStr = std::to_string(Pstream::myProcNo());

        auto inputCentersDatasetName = 
            "input_centers_dataset_MPI_" + mpiIndexStr; 
        SmartRedis::DataSet inputCentersDataset(inputCentersDatasetName);
        // Create the boundary displacements DataSet
        auto inputFieldDatasetName = 
            "input_" + field + "_dataset_MPI_" + mpiIndexStr;
        SmartRedis::DataSet inputFieldDataset(inputFieldDatasetName);

        // Add the type name into the input field dataset metadata. 
        inputFieldDataset.add_meta_string("type", "scalar");

        //- Put the cell centers into the centers dataset 
        // TODO(TM): double-chek mesh.nCells!
        Pout << "Number of cells : " << inputMesh.nCells() << endl;
        Info << "Writing cell centers to smartredis. " << endl;
        inputCentersDataset.add_tensor("input_cells_MPI_" + mpiIndexStr,
                                  (void*)inputMesh.C().cdata(), 
                                  std::vector<size_t>{size_t(inputMesh.nCells()), 3},
                                  SRTensorTypeDouble, SRMemLayoutContiguous);


        //- Put the input field in smartredis.
        Info << "Writing field " << field << " to smartredis. " << endl;
        inputFieldDataset.add_tensor("input_" + field + "_MPI_" + mpiIndexStr,
                                     (void*)inputVolScalarFieldTmp->cdata(), 
                                      std::vector<size_t>{size_t(inputMesh.nCells()), 1},
                                      SRTensorTypeDouble, SRMemLayoutContiguous);

        smartRedisClient.put_dataset(inputCentersDataset);
        smartRedisClient.put_dataset(inputFieldDataset);
        smartRedisClient.append_to_list("inputCentersDatasetList", 
                                        inputCentersDataset);
        smartRedisClient.append_to_list("inputFieldDatasetList", 
                                         inputFieldDataset);

        
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
