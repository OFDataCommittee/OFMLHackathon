/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2022 Tomislav Maric, TU Darmstadt
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
    pinnFoam

Description

\*---------------------------------------------------------------------------*/

// libtorch
#include <torch/torch.h>
#include "ATen/Functions.h"
#include "ATen/core/interned_strings.h"
#include "torch/nn/modules/activation.h"
#include "torch/optim/lbfgs.h"
#include "torch/optim/rmsprop.h"

// STL
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <filesystem>

// OpenFOAM
#include "fvCFD.H"

// libtorch-OpenFOAM data transfer
#include "torchFunctions.C"
#include "fileNameGenerator.H"

using namespace Foam;
using namespace torch::indexing;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addOption
    (
        "volFieldName",
        "string",
        "Name of the volume (cell-centered) field approximated by the neural network."
    );

    argList::addOption
    (
        "hiddenLayers",
        "int,int,int,...",
        "A sequence of hidden-layer depths."
    );

    argList::addOption
    (
        "optimizerStep",
        "double",
        "Step of the optimizer."
    );

    argList::addOption
    (
        "maxIterations",
        "<int>",
        "Max number of iterations."
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    // Initialize hyperparameters

    // - NN architecture
    DynamicList<label> hiddenLayers;
    scalar optimizerStep;
    // - Maximal number of training iterations.
    std::size_t maxIterations;

    // - Initialize hyperparameters from command line arguments if they are provided
    if (args.found("hiddenLayers") &&
        args.found("optimizerStep") &&
        args.found("maxIterations"))
    {
        hiddenLayers = args.get<DynamicList<label>>("hiddenLayers");
        optimizerStep = args.get<scalar>("optimizerStep");
        maxIterations = args.get<label>("maxIterations");
    }
    else // Initialize from system/fvSolution.AI.approximator sub-dict.
    {
        const fvSolution& fvSolutionDict (mesh);
        const dictionary& aiDict = fvSolutionDict.subDict("AI");

        hiddenLayers = aiDict.get<DynamicList<label>>("hiddenLayers");
        optimizerStep = aiDict.get<scalar>("optimizerStep");
        maxIterations = aiDict.get<label>("maxIterations");
    }

    // Use double-precision floating-point arithmetic.
    torch::set_default_dtype(
        torch::scalarTypeToTypeMeta(torch::kDouble)
    );

    // Construct the MLP
    torch::nn::Sequential nn;
    // - Input layer are always the 3 spatial coordinates in OpenFOAM, 2D
    //   simulations are pseudo-2D (single cell-layer).
    nn->push_back(torch::nn::Linear(3, hiddenLayers[0]));
    nn->push_back(torch::nn::GELU()); // FIXME: RTS activation function.
    // - Hidden layers
    for (label L=1; L < hiddenLayers.size(); ++L)
    {
        nn->push_back(
            torch::nn::Linear(hiddenLayers[L-1], hiddenLayers[L])
        );
        // TODO: RTS Alternatives TM.
        nn->push_back(torch::nn::GELU());
        //nn->push_back(torch::nn::Tanh());
    }
    // - Output is 1D: value of the learned scalar field.
    // TODO: generalize here for vector / scalar data.
    nn->push_back(
        torch::nn::Linear(hiddenLayers[hiddenLayers.size() - 1], 4)
    );

    // Initialize training data

    // - Reinterpreting OpenFOAM's fileds as torch::tensors without copying
    //  - Reinterpret OpenFOAM's input volScalarField as scalar* array
    volScalarField::pointer Phi_data = Phi.ref().data();
    volVectorField::pointer U_data = U.ref().data();
    //  - Use the scalar* (volScalarField::pointer) to view
    //    the volScalarField as torch::Tensor without copying data.
    torch::Tensor Phi_tensor = torch::from_blob(Phi_data, {Phi.size(), 1});
    Phi_tensor.requires_grad_(true);
    torch::Tensor U_tensor = torch::from_blob(U_data, {U.size(), 3});
    U_tensor.requires_grad_(true);
    torch::Tensor O_tensor = torch::hstack({U_tensor,Phi_tensor}).view({Phi.size(),4});
    O_tensor.requires_grad_(true);
    //  - Reinterpret OpenFOAM's vectorField as vector* array
    volVectorField& cc = const_cast<volVectorField&>(mesh.C());
    volVectorField::pointer cc_data = cc.ref().data();
    //  - Use the scalar* (volScalarField::pointer) to view
    //    the volScalarField as torch::Tensor without copying data.
    torch::Tensor cc_tensor = torch::from_blob(cc_data, {cc.size(),3});

    // - Randomly shuffle cell center indices.
    torch::Tensor shuffled_indices = torch::randperm(
        mesh.nCells(),
        torch::TensorOptions().dtype(at::kLong)
    );
    // - Randomly select 10 % of all cell centers for training.
    long int n_cells = int(0.1 * mesh.nCells());
    torch::Tensor training_indices = shuffled_indices.index({Slice(0, n_cells)});

    // - Use 10% of random indices to select the training_data from Phi_tensor
    torch::Tensor O_training = O_tensor.index(training_indices);
    O_training.requires_grad_(true);

    torch::Tensor cc_training = cc_tensor.index(training_indices);
    cc_training.requires_grad_(true);

    // Train the network
    torch::optim::RMSprop optimizer(nn->parameters(), optimizerStep);

    torch::Tensor O_predict = torch::zeros_like(O_training);
    O_predict.requires_grad_(true);
    torch::Tensor mse = torch::zeros_like(O_training);



    size_t epoch = 1;
    double min_mse = 1.;

    // - Approximate DELTA_X on unstructured meshes
    const auto& deltaCoeffs = mesh.deltaCoeffs().internalField();
    double delta_x = Foam::pow(
        Foam::min(deltaCoeffs).value(),-1
    );

    // - Open the data file for writing
    auto file_name = getAvailableFileName("pinnPotentialFoam");
    std::ofstream dataFile (file_name);
    dataFile << "HIDDEN_LAYERS,OPTIMIZER_STEP,MAX_ITERATIONS,"
        << "DELTA_X,EPOCH,DATA_MSE,GRAD_MSE,TRAINING_MSE\n";

    // - Initialize the best model (to be saved during training)
    torch::nn::Sequential nn_best;
    for (; epoch <= maxIterations; ++epoch)
    {
        // Training
        optimizer.zero_grad();

        // Compute the prediction from the nn.
        O_predict = nn->forward(cc_training);
    //    );


        //Jacobian vector product
        /*
        auto U_predict_grad = torch::autograd::grad(
           {O_predict.index({Slice(),Slice(0,2)})},//N_{train} x 1
           {cc_training}, // N_{train} x 3
           {torch::ones_like(O_training.index({Slice(),Slice(0,2)}))}, // N_{train} x 1
           true,
           true
        );
        */
        
        //grad(Ux) = gradient of scalar component Ux w.r.t (x,y,z)
        auto Ux_predict_grad = torch::autograd::grad(
           {O_predict.index({Slice(),0})},//N_{train} x 1
           {cc_training}, // N_{train} x 3
           {torch::ones_like(O_training.index({Slice(),0}))}, // N_{train} x 1
           true,
           true
        );
        
        //grad(Uy) = gradient of scalar component Uy w.r.t (x,y,z)
        auto Uy_predict_grad = torch::autograd::grad(
           {O_predict.index({Slice(),1})},//N_{train} x 1
           {cc_training}, // N_{train} x 3
           {torch::ones_like(O_training.index({Slice(),1}))}, // N_{train} x 1
           true,
           true
        );
                
        //grad(Uz) = gradient of scalar component Uz w.r.t (x,y,z)
        auto Uz_predict_grad = torch::autograd::grad(
           {O_predict.index({Slice(),2})},//N_{train} x 1
           {cc_training}, // N_{train} x 3
           {torch::ones_like(O_training.index({Slice(),2}))}, // N_{train} x 1
           true,
           true
        );
        
        auto divU = Ux_predict_grad[0].index({Slice(), 0}) + Uy_predict_grad[0].index({Slice(), 1}) + Uz_predict_grad[0].index({Slice(), 2});
        
        
        // grad(Phi) = gradient of the scalar potenial Phi w.r. (x,y,z)
        auto Phi_predict_grad = torch::autograd::grad(
           {O_predict.index({Slice(),3})},//N_{train} x 1
           {cc_training}, // N_{train} x 3
           {torch::ones_like(O_training.index({Slice(),3}))}, // N_{train} x 1
           true,
           true
        );
        
        
        auto Phi_predict_grad_x_grad = torch::autograd::grad(
           {Phi_predict_grad[0].index({Slice(),0})},//N_{train} x 1
           {cc_training}, // N_{train} x 3
           {torch::ones_like(Phi_predict_grad[0].index({Slice(),0}))}, // N_{train} x 1
           true,
           true
        );
        
        
         auto Phi_predict_grad_y_grad = torch::autograd::grad(
           {Phi_predict_grad[0].index({Slice(),1})},//N_{train} x 1
           {cc_training}, // N_{train} x 3
           {torch::ones_like(Phi_predict_grad[0].index({Slice(),1}))}, // N_{train} x 1
           true,
           true
        );
        
         auto Phi_predict_grad_z_grad = torch::autograd::grad(
           {Phi_predict_grad[0].index({Slice(),2})},//N_{train} x 1
           {cc_training}, // N_{train} x 3
           {torch::ones_like(Phi_predict_grad[0].index({Slice(),2}))}, // N_{train} x 1
           true,
           true
        );
        
        
        auto laplacePhi = Phi_predict_grad_x_grad[0].index({Slice(), 0}) + Phi_predict_grad_y_grad[0].index({Slice(), 1}) + Phi_predict_grad_z_grad[0].index({Slice(), 2});
        // Compute the data mse loss.
        auto mse_data = mse_loss(O_predict, O_training);
      //  );
        
        // div.grad(Phi) - div.U = 0
        
        auto potentialEqnResidual = laplacePhi - divU;
        
        auto mse_grad = mse_loss(
            potentialEqnResidual, 
            torch::zeros_like(O_training.index({Slice(), 0}))
        );
        
        // Combine the losses into a Physics Informed Neural Network.
        mse = mse_data + mse_grad;
        
        // Optimize weights of the PiNN.
        mse.backward({},true);
        optimizer.step();

        std::cout << "Epoch = " << epoch << "\n"
            << "DATA_MSE = " << mse_data.item<double>() << "\n"
  //        << "U MSE = " << mse_U.item<double>() << "\n"
            << "Training MSE = " << mse.item<double>() << "\n";


        std::cout << at::size(Ux_predict_grad[0],0) << at::size(Uy_predict_grad[0],0) << at::size(Uz_predict_grad[0],0) << at::size(Phi_predict_grad[0],0) << "\n";
        // Write the hiddenLayers_ network structure as a string-formatted python list.
        
        std::cout << at::size(divU,0) << "\n";
        std::cout << at::size(laplacePhi,0) << "\n";
        dataFile << "\"";
        for(decltype(hiddenLayers.size()) i = 0; i < hiddenLayers.size() - 1; ++i)
            dataFile << hiddenLayers[i] << ",";
        dataFile  << hiddenLayers[hiddenLayers.size() - 1]
            << "\"" << ",";
        // Write the rest of the data.
        dataFile << optimizerStep << "," << maxIterations << ","
            << delta_x << "," << epoch << ","
            << mse_data.item<double>() << ","
        //    << mse_U.item<double>() << ","
            << mse.item<double>() << std::endl;

        if (mse.item<double>() < min_mse)
        {
            min_mse = mse.item<double>();
            // Save the "best" model with the minimal MSE over all epochs.
            nn_best = nn;
        }
    }

    // Evaluate the best NN.
    //  - Reinterpret OpenFOAM's output volScalarField as scalar* array
    volScalarField::pointer Phi_nn_data = Phi_nn.ref().data();
    volVectorField::pointer U_nn_data = U_nn.ref().data();
    //  - Use the scalar* (volScalarField::pointer) to view
    //    the volScalarField as torch::Tensor without copying data.
    //volScalarField::Tensor Phi_nn_tensor = torch::from_blob(Phi_nn_data, {Phi.size()});
    torch::Tensor Phi_nn_tensor = torch::from_blob(Phi_nn_data, {Phi.size()});
    //volVectorField::Tensor U_nn_tensor = torch::from_blob(U_nn_data, {U.size()});
    torch::Tensor U_nn_tensor = torch::from_blob(U_nn_data, {U.size()});
    //  - Evaluate the volumeScalarField vf_nn using the best NN model.
    torch::Tensor O_nn_tensor = torch::from_blob(Phi_nn_data, {Phi.size()});
    O_nn_tensor = nn_best->forward(cc_tensor);
    //  - FIXME: 2022-06-01, the C++ PyTorch API does not overwrite the blob object.
    //           If a Model is coded by inheritance, maybe forward(input, output) is
    //           available, that overwrites the data in vf_nn by acting on the
    //           non-const view of the data giv cellIen by vf_nn_tensor. TM.
    forAll(Phi_nn, cellI)
    {
        Phi_nn[cellI] = O_nn_tensor[cellI][3].item<double>();
    }

    forAll(U_nn, cellI)
    {
        U_nn[cellI][0] = O_nn_tensor[cellI][0].item<double>();
        U_nn[cellI][1] = O_nn_tensor[cellI][1].item<double>();
        U_nn[cellI][2] = O_nn_tensor[cellI][2].item<double>();
    }

    //  - Evaluate the vf_nn boundary conditions.
    Phi_nn.correctBoundaryConditions();
    U_nn.correctBoundaryConditions();
    // Error calculation and output.
    // - Data

    error_U == Foam::mag(U - U_nn);
    scalar error_U_inf = Foam::max(error_U).value();
    scalar error_U_mean = Foam::average(error_U).value();

    error_Phi == Foam::mag(Phi - Phi_nn);
    scalar error_Phi_inf = Foam::max(error_Phi).value();
    scalar error_Phi_mean = Foam::average(error_Phi).value();



    // - Gradient
 //   volVectorField vf_grad ("vf_grad", fvc::grad(vf));
  //  volVectorField vf_nn_grad ("vf_nn_grad", fvc::grad(vf_nn));
  //  volScalarField error_grad_c ("error_grad_c", Foam::mag(vf_grad - vf_nn_grad));


    Info << "max(|field - field_nn|) = " << error_Phi_inf << endl;
    Info << "mean(|field - field_nn|) = " << error_Phi_mean << endl;
    Info << "max(|field - field_nn|) = " << error_U_inf << endl;
    Info << "mean(|field - field_nn|) = " << error_U_mean << endl;


    // Write fields
    error_Phi.write();
    Phi_nn.write();
    error_U.write();
    U_nn.write();
    //vf_nn_grad.write();
    //vf_grad.write();
    //error_grad_c.write();

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
