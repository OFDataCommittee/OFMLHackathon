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

    Uses SmartSim/SmartRedis to train an ML model on a field on a coarse mesh
    resolution and then approximates the field on the fine mesh resolution
    using the ML model.

\*---------------------------------------------------------------------------*/


// Initialize the input time  

fileName inputCase = args.get<fileName>("inputCase"); 
inputCase.clean();
const fileName inputRoot = inputCase.path();
inputCase = inputCase.name();

Info << "Input case: " << inputCase << endl;
Time runTimeInput 
(
    Time::controlDictName,
    inputRoot,
    inputCase 
);


// Initialize the output case time  

fileName outputCase = args.get<fileName>("outputCase"); 
outputCase.clean(); 
const fileName outputRoot = outputCase.path();
outputCase = outputCase.name(); 

Info << "Output case: " << outputCase << endl;
Time runTimeOutput 
(
    Time::controlDictName,
    outputRoot,
    outputCase 
);

// ************************************************************************* //
