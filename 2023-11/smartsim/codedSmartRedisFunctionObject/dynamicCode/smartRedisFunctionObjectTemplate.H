/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019 OpenCFD Ltd.
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

Description
    Template for use with dynamic code generation of a functionObject.

SourceFiles
    functionObjectTemplate.C

\*---------------------------------------------------------------------------*/

#ifndef coded_functionObjectTemplate_H
#define coded_functionObjectTemplate_H

#include "fvMesh.H"
#include "fvMeshFunctionObject.H"
#include "dictionaryContent.H"
#include "smartRedisAdapter.H"

//{{{ begin codeInclude
${codeInclude}
//}}} end codeInclude

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward Declarations
class fvMesh;

/*---------------------------------------------------------------------------*\
                         A templated functionObject
\*---------------------------------------------------------------------------*/

class ${typeName}FunctionObject
:
    public functionObjects::fvMeshFunctionObject,
    public dictionaryContent
{
    // Private Data
    
        //- Client name (for debug output)
        word clientName_;

        //- The RedisAI database client
        tmp<smartRedisAdapter> redisDB_;

//{{{ begin codeData
        ${codeData}
//}}} end codeData


    // Private Member Functions

        //- Report a message with the SHA1sum
        inline static void printMessage(const char* message)
        {
            Info<< message << " sha1: " << SHA1sum << '\n';
        }

        //- Cast reference of objectRegistry to fvMesh
        const fvMesh& mesh() const;

        //- No copy construct
        ${typeName}FunctionObject
        (
            const ${typeName}FunctionObject&
        ) = delete;

        //- No copy assignment
        void operator=
        (
            const ${typeName}FunctionObject&
        ) = delete;

public:

    //- SHA1 representation of the code content
    static constexpr const char* const SHA1sum = "${SHA1sum}";

    //- Runtime type information
    TypeName("${typeName}");

    template<class cmptType>
    struct NComponents {
        static constexpr label value = cmptType::nComponents;
    };

    // Constructors

        //- Construct from Time and dictionary
        ${typeName}FunctionObject
        (
            const word& name,
            const Time& runTime,
            const dictionary& dict
        );


    //- Destructor
    virtual ~${typeName}FunctionObject();


    // Member Functions

        //- Code context as a dictionary
        const dictionary& codeContext() const noexcept
        {
            return dictionaryContent::dict();
        }

        //- Read optional controls
        virtual bool read(const dictionary& dict);

        //- Execute (at time-step)
        virtual bool execute();

        //- Write (at write interval)
        virtual bool write();

        //- Executed at the final time-loop
        virtual bool end();

        //- Send fields to SmartRedis
        template<class T>
        void sendField(const word& fieldName) {
            const T& sField = mesh().lookupObject<T>(fieldName);
            std::vector<size_t> dims = {size_t(sField.size()), size_t(NComponents<typename T::cmptType>::value)};
            redisDB_->client().put_tensor
            (
                sField.name(),
                (void*)sField.internalField().cdata(),
                dims,
                SRTensorTypeDouble, SRMemLayoutContiguous
            );
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
