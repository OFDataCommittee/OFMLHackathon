/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2021 Tomislav Maric, TU Darmstadt
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

    Generates a file name of the format baseName-00000N.extension if the
    file does not already exist.

\*---------------------------------------------------------------------------*/

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <filesystem>
#include <fstream>

std::string getAvailableFileName
(
    const std::string baseName,
    const std::string extension=".csv"
)
{
    using namespace std;
    namespace fs = std::filesystem;

    int i = 0;
    stringstream ss;
    do
    {
        ss.str(std::string()); // Clears the string stream.
        ss << baseName // Name the data file using typeName.
            << "-" << setw(8) << setfill('0') << i // Pad the unique data file ID with 0s.
            << extension;
        fs::path f {ss.str()};
        ++i;
        if (!fs::exists(f))
        {
            break;
        }
    } while(true);

    return ss.str();
}
