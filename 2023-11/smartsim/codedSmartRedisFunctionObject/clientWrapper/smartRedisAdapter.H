/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2023 AUTHOR, AFFILIATION
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
    Foam::smartRedisAdapter

Description

\*---------------------------------------------------------------------------*/

#ifndef smartRedisAdapter_H
#define smartRedisAdapter_H

#include "regIOobject.H"
#include "client.h"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                   Class smartRedisAdapter Declaration
\*---------------------------------------------------------------------------*/

class smartRedisAdapter
:
    public regIOobject,
    public refCount
{
protected:

    // Protected Data
        
        //- cluster mode
        bool clusterMode_;

        //- SmartRedis Database Client
        SmartRedis::Client client_;

public:

    //- Runtime type information
    TypeName("smartRedisAdapter");


    // Constructors

        //- Construct from Time and dictionary
        smartRedisAdapter
        (
            const IOobject& io,
            const dictionary& dict
        );

        //- Construct from a pointer by moving the client
        explicit smartRedisAdapter
        (
            smartRedisAdapter* ptr
        );

        //- No copy construct
        smartRedisAdapter(const smartRedisAdapter&) = delete;

        //- No copy assignment
        void operator=(const smartRedisAdapter&) = delete;


    //- Destructor
    virtual ~smartRedisAdapter() = default;


    // Member Functions
    
        //- return the client's instance
        SmartRedis::Client& client() { return client_; };

        //- Implement writing to ostream from regIOobject
        virtual bool writeData(Ostream&) const { return true; }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
