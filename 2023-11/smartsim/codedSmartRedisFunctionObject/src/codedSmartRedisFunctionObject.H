/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2021 OpenCFD Ltd.
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
    Foam::functionObjects::codedSmartRedisFunctionObject

Group
    grpUtilitiesFunctionObjects

Description
    Provides a general interface to enable dynamic code compilation.

    The entries are:
    \plaintable
       codeInclude   | include files
       codeOptions   | compiler line: added to EXE_INC (Make/options)
       codeLibs      | linker line: added to LIB_LIBS (Make/options)
       codeData      | c++; local member data (default constructed by default)
       codeConstruct | c++; local member data (initialization list for member data)
       localCode     | c++; local static functions
       codeRead      | c++; upon functionObject::read()
       codeExecute   | c++; upon functionObject::execute()
       codeWrite     | c++; upon functionObject::write()
       codeEnd       | c++; upon functionObject::end()
       codeContext   | additional dictionary context for the code
    \endplaintable

Usage
    Example of function object specification:
    \verbatim
    difference
    {
        type coded;
        libs (utilityFunctionObjects);

        // Name of on-the-fly generated functionObject
        name writeMagU;
        codeWrite
        #{
            // Lookup U
            const volVectorField& U = mesh().lookupObject<volVectorField>("U");

            // Write
            mag(U)().write();
        #};
    }
    \endverbatim

Note
    The code context dictionary can be supplied separately as the
    \c codeContext entry.

See also
    Foam::functionObject
    Foam::codedBase

SourceFiles
    codedSmartRedisFunctionObject.C

\*---------------------------------------------------------------------------*/

#ifndef functionObjects_codedSmartRedisFunctionObject_H
#define functionObjects_codedSmartRedisFunctionObject_H

#include "timeFunctionObject.H"
#include "codedBase.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{

/*---------------------------------------------------------------------------*\
                     Class codedSmartRedisFunctionObject Declaration
\*---------------------------------------------------------------------------*/

class codedSmartRedisFunctionObject
:
    public functionObjects::timeFunctionObject,
    public codedBase
{
protected:

    // Protected Data

        //- Input dictionary
        dictionary dict_;

        word name_;

        string codeData_;
        string codeRead_;
        string codeConstruct_;
        string codeExecute_;
        string codeWrite_;
        string codeEnd_;

        //- Underlying functionObject
        mutable autoPtr<functionObject> redirectFunctionObjectPtr_;


    // Protected Member Functions

        //- Mutable access to the loaded dynamic libraries
        virtual dlLibraryTable& libs() const;

        //- Description (type + name) for the output
        virtual string description() const;

        //- Clear redirected object(s)
        virtual void clearRedirect() const;

        //- Additional 'codeContext' dictionary to pass through
        virtual const dictionary& codeContext() const;

        //- The code dictionary
        virtual const dictionary& codeDict() const;

        //- Adapt the context for the current object
        virtual void prepare(dynamicCode&, const dynamicCodeContext&) const;


        //- No copy construct
        codedSmartRedisFunctionObject(const codedSmartRedisFunctionObject&) = delete;

        //- No copy assignment
        void operator=(const codedSmartRedisFunctionObject&) = delete;


public:

    // Static Data Members

        //- Name of the C code template to be used
        static constexpr const char* const codeTemplateC
            = "smartRedisFunctionObjectTemplate.C";

        //- Name of the H code template to be used
        static constexpr const char* const codeTemplateH
            = "smartRedisFunctionObjectTemplate.H";


    //- Runtime type information
    TypeName("coded");


    // Constructors

        //- Construct from Time and dictionary
        codedSmartRedisFunctionObject
        (
            const word& name,
            const Time& runTime,
            const dictionary& dict
        );


    //- Destructor
    virtual ~codedSmartRedisFunctionObject() = default;


    // Member Functions

        //- Dynamically compiled functionObject
        functionObject& redirectFunctionObject() const;

        //- Called at each ++ or += of the time-loop.
        //  postProcess overrides the usual executeControl behaviour and
        //  forces execution (used in post-processing mode)
        virtual bool execute();

        //- Called at each ++ or += of the time-loop.
        //  postProcess overrides the usual writeControl behaviour and
        //  forces writing always (used in post-processing mode)
        virtual bool write();

        //- Called when Time::run() determines that the time-loop exits.
        //  By default it simply calls execute().
        virtual bool end();

        //- Read and set the function object if its data have changed
        virtual bool read(const dictionary&);
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
