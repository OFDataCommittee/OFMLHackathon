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

#include "smartRedisAdapter.H"
#include "Time.H"


// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(smartRedisAdapter, 0);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::smartRedisAdapter::smartRedisAdapter
(
    const IOobject& io,
    const dictionary& dict
)
:
    regIOobject(io),
    refCount(),
    clusterMode_(dict.getOrDefault<Switch>("clusterMode", true)),
    client_(clusterMode_, io.name()) // deprecated constructor though
{
}

Foam::smartRedisAdapter::smartRedisAdapter
(
    smartRedisAdapter* ptr
)
:
    regIOobject(*ptr),
    refCount(),
    clusterMode_(ptr->clusterMode_),
    client_(std::move(ptr->client_)) // no copy of the client
{
}


// ************************************************************************* //
