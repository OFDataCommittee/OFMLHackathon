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

#include "IOdictionary.H"
#include "objectRegistry.H"
#include "smartSimFunctionObject.H"
#include "Time.H"
#include "fvMesh.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"

#include <vector>
#include <string>


// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(smartSimFunctionObject, 0);
    addToRunTimeSelectionTable(functionObject, smartSimFunctionObject, dictionary);
    SmartRedis::Client smartSimFunctionObject::redisDB(false, "default");
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
    //clusterMode_(dict.getOrDefault<bool>("clusterMode", true)),
    fieldNames_(dict.get<wordList>("fieldNames")),
    fieldDimensions_(dict.get<labelList>("fieldDimensions"))//,
    //client_(clusterMode_)
{
    read(dict);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::functionObjects::smartSimFunctionObject::read(const dictionary& dict)
{
    // TODO(TM): read the model from SmartRedis, initialize the fields from the model.
    return true;
}

bool Foam::functionObjects::smartSimFunctionObject::execute()
{
    if (time_.timeIndex() == 1)
    {
        // TODO(TM): 
        // - get non-const refs to the fields.
        // - for all fields 
        //  - read the field model from the database 
        //  - overwrite the field with the field model
    }

    return true;
}

bool Foam::functionObjects::smartSimFunctionObject::end()
{
    Info << "Fields: " << fieldNames_ << endl;
    Info << "Field dimensions: " << fieldDimensions_ << endl;
    
    if (fieldNames_.size() != fieldDimensions_.size())
    {
        FatalErrorInFunction
            << "fieldNames and fieldDimensions of different sizes"  
            << abort(FatalError);
    }

    forAll(fieldNames_, fieldI)
    {
        // Set field dimensions 
        // - nCells x 1 for a scalar field 
        // - nCells x 3 for a vector field 
        // - nCells x 6 for a symmTensor field 
        std::vector<size_t> dims = {size_t(mesh_.nCells()), 
                                    size_t(fieldDimensions_[fieldI])};
        
        if(fieldDimensions_[fieldI] == 1) // scalar field
        {
            // Get the cell-centered scalar field from the mesh (registry).
            const volScalarField& sField = mesh_.lookupObject<volScalarField>(fieldNames_[fieldI]);

            // Send the cell-centered scalar field to SmartRedis
            redisDB.put_tensor(sField.name(), (void*)sField.internalField().cdata(), dims,
                               SRTensorTypeDouble, SRMemLayoutContiguous);

        } 
        else if (fieldDimensions_[fieldI] == 3) // vector field
        {
            // Get the cell-centered scalar field from the mesh (registry).
            const volVectorField& vField = mesh_.lookupObject<volVectorField>(fieldNames_[fieldI]);

            // Send the cell-centered scalar field to SmartRedis
            redisDB.put_tensor(vField.name(), (void*)vField.internalField().cdata(), dims,
                               SRTensorTypeDouble, SRMemLayoutContiguous);
        }
        else if (fieldDimensions_[fieldI] == 6) // TODO(TM): symmTensor field
        {
            FatalErrorInFunction
                << "Symmetric tensor field not implemented."  
                << abort(FatalError);
        }
        else
        {
            FatalErrorInFunction
                << "Unsupported field dimension, fieldDimensions[" 
                <<  fieldI << "] = " << fieldDimensions_[fieldI] 
                << abort(FatalError);
        }
    }

    return true;
}

bool Foam::functionObjects::smartSimFunctionObject::write()
{
    return true;
}


// ************************************************************************* //
