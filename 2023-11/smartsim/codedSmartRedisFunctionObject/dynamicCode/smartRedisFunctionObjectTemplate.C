/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019-2021 OpenCFD Ltd.
    Copyright (C) YEAR AUTHOR, AFFILIATION
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

#include "smartRedisFunctionObjectTemplate.H"
#define namespaceFoam  // Suppress <using namespace Foam;>
#include "fvCFD.H"
#include "fvMesh.H"
#include "unitConversion.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(${typeName}FunctionObject, 0);

addRemovableToRunTimeSelectionTable
(
    functionObject,
    ${typeName}FunctionObject,
    dictionary
);


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

// dynamicCode:
// SHA1 = ${SHA1sum}
//
// unique function name that can be checked if the correct library version
// has been loaded
extern "C" void ${typeName}_${SHA1sum}(bool load)
{
    if (load)
    {
        // Code that can be explicitly executed after loading
    }
    else
    {
        // Code that can be explicitly executed before unloading
    }
}


// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode
${localCode}
//}}} end localCode

} // End namespace Foam


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

const Foam::fvMesh&
Foam::${typeName}FunctionObject::mesh() const
{
    return refCast<const fvMesh>(obr_);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::
${typeName}FunctionObject::
${typeName}FunctionObject
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    functionObjects::fvMeshFunctionObject(name, runTime, dict),
    clientName_(dict.getOrDefault<word>("clientName", "default")),
    redisDB_(
        runTime.foundObject<smartRedisAdapter>(clientName_)
        ? &runTime.lookupObjectRef<smartRedisAdapter>(clientName_)
        : new smartRedisAdapter
            (
                IOobject
                (
                    clientName_,
                    runTime.timeName(),
                    runTime,
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                dict
            )
    )
//{{{ begin codeConstruct
        ${codeConstruct}
//}}} end codeConstruct
{
    read(dict);
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::
${typeName}FunctionObject::
~${typeName}FunctionObject()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool
Foam::
${typeName}FunctionObject::read(const dictionary& dict)
{
    if (${verbose:-false})
    {
        printMessage("read ${typeName}");
    }
    auto& client = redisDB_.ref().client();

//{{{ begin code
    ${codeRead}
//}}} end code

    return true;
}


bool
Foam::
${typeName}FunctionObject::execute()
{
    if (${verbose:-false})
    {
        printMessage("execute ${typeName}");
    }
    auto& client = redisDB_.ref().client();

//{{{ begin code
    ${codeExecute}
//}}} end code

    return true;
}


bool
Foam::
${typeName}FunctionObject::write()
{
    if (${verbose:-false})
    {
        printMessage("write ${typeName}");
    }
    auto& client = redisDB_.ref().client();

//{{{ begin code
    ${codeWrite}
//}}} end code

    return true;
}


bool
Foam::
${typeName}FunctionObject::end()
{
    if (${verbose:-false})
    {
        printMessage("end ${typeName}");
    }
    auto& client = redisDB_.ref().client();

//{{{ begin code
    ${codeEnd}
//}}} end code

    return true;
}

// Specialization components for double
template<>
struct Foam::${typeName}FunctionObject::NComponents<double>
{
    static constexpr label value = 1;
};

// ************************************************************************* //
