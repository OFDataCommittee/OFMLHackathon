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
    foamSmartSimDmd

Description

    Uses SmartSim/SmartRedis for streaming DMD of OpenFOAM fields.

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
    // Selecting the time step folders for the approximated fields.
    timeSelector::addOptions();

    // Add the option to the application for the name of the approximated field.
    argList::addOption
    (
        "fieldName", 
        "fieldName",
        "Name of the approximated field, e.g. p." 
    ); 

    // OpenFOAM boilerplate: set root folder and options
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    // Read the name of the approximated field from the command line option
    // foamSmartSimDmd -case cavity -fieldName p
    const word fieldName = args.get<word>("fieldName");
    Info << "Approximating field " << fieldName << endl;

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

        auto inputFieldDatasetName = 
            "input_" + fieldName + "_dataset_MPI_" + mpiIndexStr;
        SmartRedis::DataSet inputFieldDataset(inputFieldDatasetName);

        // Add the type name into the input field dataset metadata. 
        inputFieldDataset.add_meta_string("type", "scalar");

        //- Put the input field in smartredis.
        Info << "Writing field " << fieldName << " to smartredis. " << endl;
        inputFieldDataset.add_tensor("input_" + fieldName + "_MPI_" + mpiIndexStr,
                                     (void*)inputVolScalarFieldTmp->cdata(), 
                                      std::vector<size_t>{size_t(mesh.nCells()), 1},
                                      SRTensorTypeDouble, SRMemLayoutContiguous);

        smartRedisClient.put_dataset(inputFieldDataset);
        smartRedisClient.append_to_list("inputFieldDatasetList", 
                                         inputFieldDataset);
    }
    // If 'field' is a volVectorField 
    else if (!inputVolVectorFieldTmp->empty())
    {
        // TODO(TM): same as for the volScalarField only with vector dimensions.
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    Info<< nl;
    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
