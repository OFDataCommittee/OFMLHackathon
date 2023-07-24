/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2023 Tomislav Maric, TU Darmstadt 
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

#include "Pstream.H"
#include "displacementSmartSimMotionSolver.H"
#include "addToRunTimeSelectionTable.H"
#include "OFstream.H"
#include "meshTools.H"
#include "mapPolyMesh.H"
#include "fvPatch.H"
#include "fixedValuePointPatchFields.H"
//#include "mpi.h"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(displacementSmartSimMotionSolver, 0);

    addToRunTimeSelectionTable
    (
        motionSolver,
        displacementSmartSimMotionSolver,
        dictionary
    );

    addToRunTimeSelectionTable
    (
        displacementMotionSolver,
        displacementSmartSimMotionSolver,
        displacement
    );
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::displacementSmartSimMotionSolver::displacementSmartSimMotionSolver
(
    const polyMesh& mesh,
    const IOdictionary& dict
)
:
    displacementMotionSolver(mesh, dict, typeName),
    clusterMode_(this->coeffDict().get<bool>("clusterMode")), 
    client_(clusterMode_)
{}

Foam::displacementSmartSimMotionSolver::
displacementSmartSimMotionSolver
(
    const polyMesh& mesh,
    const IOdictionary& dict,
    const pointVectorField& pointDisplacement,
    const pointIOField& points0
)
:
    displacementMotionSolver(mesh, dict, pointDisplacement, points0, typeName),
    clusterMode_(dict.getOrDefault<bool>("clusterMode", true)),
    client_(clusterMode_)
{}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::displacementSmartSimMotionSolver::
~displacementSmartSimMotionSolver()
{}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::pointField> Foam::displacementSmartSimMotionSolver::curPoints() const
{
    return mesh().points() + pointDisplacement(); 
}

void Foam::displacementSmartSimMotionSolver::solve()
{
    // Apply boundary conditions to point displacements.    
    pointDisplacement_.boundaryFieldRef().updateCoeffs();

    // Send mesh boundary points and displacements to smartRedis 
    const auto& boundaryDisplacements = 
        pointDisplacement().boundaryField();
    const auto& meshBoundary = mesh().boundaryMesh(); 

    // Time step and MPI rank used for identifying data
    const auto& runTime = mesh().time();
    label timeIndex = runTime.timeIndex();
    auto timeIndexStr = std::to_string(timeIndex);
    auto mpiIndexStr = std::to_string(Pstream::myProcNo());

    // Write the current time index to SmartRedis
    //if (Pstream::myProcNo() == 0)
    //{
        //std::vector<double> time_index_value {double(timeIndex)};
        //client_.put_tensor("time_index", time_index_value.data(), {1}, 
                            //SRTensorTypeDouble, SRMemLayoutContiguous);
    //}

    // Create the boundary points DataSet
    auto pointsDsetName = "points_dataset_timeStep_" + timeIndexStr +
        "_MPI_" + mpiIndexStr; 
    SmartRedis::DataSet pointsDataset(pointsDsetName);
    // Create the boundary displacements DataSet
    auto displDsetName = "displacement_dataset_timeStep_" + timeIndexStr +  
        "_MPI_" + mpiIndexStr;
    SmartRedis::DataSet displacementsDataset(displDsetName);

    // Write the current time index to SmartRedis
    //if (Pstream::myProcNo() == 0)
    //{
        //std::vector<double> time_index_value {double(timeIndex)};
        //client_.put_tensor("time_index", time_index_value.data(), {1}, 
                            //SRTensorTypeDouble, SRMemLayoutContiguous);
    //}

    forAll(boundaryDisplacements, patchI)
    {
        // FIXME(TM): use isA<>, not hardcoded names.
        if ((meshBoundary[patchI].type() == "empty") || 
            (meshBoundary[patchI].type() == "processor"))
        {
            Pout << "Skipping " << meshBoundary[patchI].name() << ", "
                << meshBoundary[patchI].type() << endl;
            continue;
        }
        
        const polyPatch& patch = meshBoundary[patchI];

        const pointField& patchPoints = patch.localPoints();
        const pointPatchVectorField& patchDisplacements = 
            boundaryDisplacements[patchI];
        vectorField patchDisplacementData = patchDisplacements.patchInternalField(); 

        // Point patch addressing is global - the boundary loop on each MPI rank
        // sees all patches, and those not available in this MPI rank will have 
        // size 0. Size 0 data cannot be written into the SmartRedis database.
        if (patch.size() == 0)
        {
            Pout << "Skipping " << patch.name() << " with points size "
                << patchPoints.size() << " and displacements size " 
                << patchDisplacementData.size() << endl;
            continue;
        }
    
        Pout << "Sending " << patch.name() 
             << "points size " << patchPoints.size() << endl
             << " displacements size " << patchDisplacementData.size() << endl
             << " to SmartRedis." << endl;
        
        // Add the patch points to the boundary points dataset 
        auto pointsName = "points_" + patch.name() + 
            "_timeStep_" + timeIndexStr + "_MPI_" + mpiIndexStr; 
        pointsDataset.add_tensor(pointsName,
                                 (void*)patchPoints.cdata(), 
                                 std::vector<size_t>{size_t(patchPoints.size()), 3},
                                 SRTensorTypeDouble, SRMemLayoutContiguous);

        // Add the patch displacements to the boundary displacements dataset 
        auto displacementsName = "displacements_" + patch.name() +
            "_timeStep_" + timeIndexStr + "_MPI_" + mpiIndexStr;  
        displacementsDataset.add_tensor(displacementsName,
                                        (void*)patchDisplacementData.cdata(), 
                                        std::vector<size_t>{size_t(patchPoints.size()), 3},
                                        SRTensorTypeDouble, SRMemLayoutContiguous);
        
    }

    client_.put_dataset(pointsDataset);
    client_.put_dataset(displacementsDataset);
    client_.append_to_list("pointsDatasetList", pointsDataset);
    client_.append_to_list("displacementsDatasetList", displacementsDataset);

    bool model_updated = client_.poll_key("model_updated", 10, 1000);
    if (! model_updated)
    {
        FatalErrorInFunction
            << "Displacement model not found in SmartRedis database."
            << exit(Foam::FatalError);
    }
    else
    {
        std::vector<double> model_index {1};
        client_.unpack_tensor("model_updated", model_index.data(), {1},
                              SRTensorTypeDouble, SRMemLayoutContiguous);
        Pout << "Model from time step " << model_index.size() 
            << " " << model_index[0]  << endl;
    }

    // TODO(TM): hardcoded, make it work with runTime.end(). 
    if ((Pstream::myProcNo() == 0) && (timeIndex == 3))
    {
        std::vector<double> end_time_vec {double(timeIndex)};
        Info << "Seting end time flag : " << end_time_vec[0] << endl;
        client_.put_tensor("end_time_index", end_time_vec.data(), {1}, 
                            SRTensorTypeDouble, SRMemLayoutContiguous);
    }

    // Emulate MPI_Barrier()
    label totalRank = Pstream::myProcNo();
    reduce(totalRank, sumOp<label>(), totalRank);

    // Delete the model flag. 
    if (Pstream::myProcNo() == 0)
    {
        client_.delete_tensor("model_updated");
    }
}

// ************************************************************************* //
