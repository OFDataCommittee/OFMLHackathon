/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2023 AUTHOR,AFFILIATION
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

\*---------------------------------------------------------------------------*/

#include "smartSimFunctionObject.H"
#include "Time.H"
#include "fvMesh.H"
#include "addToRunTimeSelectionTable.H"

#include <vector>
#include <string>


// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(smartSimFunctionObject, 0);
    addToRunTimeSelectionTable(functionObject, smartSimFunctionObject, dictionary);
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::functionObjects::smartSimFunctionObject::smartSimFunctionObject
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    clusterMode_(dict.getOrDefault<bool>("clusterMode", true)),
    client_(clusterMode_)
{
    read(dict);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::functionObjects::smartSimFunctionObject::read(const dictionary& dict)
{
    dict.readEntry("clusterMode", clusterMode_);
    return true;
}


bool Foam::functionObjects::smartSimFunctionObject::execute()
{
    Info << "Executing SmartSim FunctionObject" << endl;
    
    // Initialize tensor dimensions
    size_t dim1 = 3;
    size_t dim2 = 2;
    size_t dim3 = 5;
    std::vector<size_t> dims = {3, 2, 5};

    // Initialize a tensor to random values.  Note that a dynamically
    // allocated tensor via malloc is also useable with the client
    // API.  The std::vector is used here for brevity.
    size_t n_values = dim1 * dim2 * dim3;
    std::vector<double> input_tensor(n_values, 0);
    for(size_t i=0; i<n_values; i++)
        input_tensor[i] = 2.0*rand()/RAND_MAX - 1.0;

    // Put the tensor in the database
    std::string key = "3d_tensor";
    client_.put_tensor(key, input_tensor.data(), dims,
                      SRTensorTypeDouble, SRMemLayoutContiguous);

    // Retrieve the tensor from the database using the unpack feature.
    std::vector<double> unpack_tensor(n_values, 0);
    client_.unpack_tensor(key, unpack_tensor.data(), {n_values},
                        SRTensorTypeDouble, SRMemLayoutContiguous);

    // Print the values retrieved with the unpack feature
    std::cout<<"Comparison of the sent and "\
                "retrieved (via unpack) values: "<<std::endl;
    for(size_t i=0; i<n_values; i++)
        std::cout<<"Sent: "<<input_tensor[i]<<" "
                 <<"Received: "<<unpack_tensor[i]<<std::endl;


    // Retrieve the tensor from the database using the get feature.
    SRTensorType get_type;
    std::vector<size_t> get_dims;
    void* get_tensor;
    client_.get_tensor(key, get_tensor, get_dims, get_type, SRMemLayoutNested);

    // Print the values retrieved with the unpack feature
    std::cout<<"Comparison of the sent and "\
                "retrieved (via get) values: "<<std::endl;
    for(size_t i=0, c=0; i<dims[0]; i++)
        for(size_t j=0; j<dims[1]; j++)
            for(size_t k=0; k<dims[2]; k++, c++) {
                std::cout<<"Sent: "<<input_tensor[c]<<" "
                         <<"Received: "
                         <<((double***)get_tensor)[i][j][k]<<std::endl;
    }

    return true;
}


bool Foam::functionObjects::smartSimFunctionObject::end()
{
    Info << "Ending SmartSim FunctionObject Execution" << endl;
    return true;
}


bool Foam::functionObjects::smartSimFunctionObject::write()
{
    Info << "Writing Data From SmartSim FunctionObject" << endl;
    return true;
}


// ************************************************************************* //
