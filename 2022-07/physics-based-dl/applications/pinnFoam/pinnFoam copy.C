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
#include <torch/torch.h> // Basic torch library and allied modules? Refer to documentation
#include "ATen/Functions.h" // General class of functions???Where is it used?
#include "ATen/core/interned_strings.h"
#include "torch/nn/modules/activation.h" //Loads activation function modules
#include "torch/optim/lbfgs.h" // Loads the lbfgs optimiser 
#include "torch/optim/rmsprop.h" //loads the rms prop optimiser

// STL - Standard template libraries
#include <algorithm>
#include <random> 
#include <numeric>
#include <cmath>
#include <filesystem>

// OpenFOAM - finite volume functionalities
#include "fvCFD.H" 

// libtorch-OpenFOAM data transfer
#include "torchFunctions.C"
#include "fileNameGenerator.H"

using namespace Foam;
using namespace torch::indexing;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// C++ program to print dimensions of a
// multidimensional array
#include <iostream>

/* 
template <typename T, size_t N>
void printDimensions(const T (&a)[N])
{
    std::cout << N;
}
 
template <typename T, size_t N, size_t M>
void printDimensions(const T (&a)[N][M])
{
    std::cout << N << "x";
    printDimensions(a[0]);
}
*/
int main(int argc, char *argv[])
{
    // Here this is to declare the variables that would be predicted by the network. 
    argList::addOption
    (
        "volFieldName",
        "string",
        "Name of the volume (cell-centered) field approximated by the neural network."
    );
    // Architecture configurations
    argList::addOption
    (
        "hiddenLayers",
        "int,int,int,...",
        "A sequence of hidden-layer depths."
    );

    // Prescribes the learning rate
    argList::addOption
    (
        "optimizerStep",
        "double",
        "Step of the optimizer."
    );
    // Maximum number of iterations
    argList::addOption
    (
        "maxIterations",
        "<int>",
        "Max number of iterations."
    );
    
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H" // To create the base mesh related information
    
    #include "createFields.H" // TO declare the fields that are being read/ computed or written

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    /******************************************************/
    // Initialize hyperparameters 
    /******************************************************/

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
    
    /***************************************************************/
    // Construct the MLP 
    /***************************************************************/
    torch::nn::Sequential nn;
    // Note:- Input layer are always the 3 spatial coordinates in OpenFOAM, 
    //   Hence, 2D simulations are pseudo-2D (single cell-layer).

    // INPUT LAYER
    nn->push_back(torch::nn::Linear(3, hiddenLayers[0])); 
    nn->push_back(torch::nn::GELU()); // FIXME: RTS activation function.
    
    // HIDDEN LAYER
    for (label L=1; L < hiddenLayers.size(); ++L)
    {
        nn->push_back(
            torch::nn::Linear(hiddenLayers[L-1], hiddenLayers[L])
        );
        // TODO: RTS Alternatives TM.
        nn->push_back(torch::nn::GELU()); 
        //nn->push_back(torch::nn::Tanh()); 
    }
    // OUTPUT LAYER
    // Note:- Output is 1D: value of the learned scalar field. 
    // TODO: generalize here for vector / scalar data. [Necessary for the potential solver]
    nn->push_back(
        torch::nn::Linear(hiddenLayers[hiddenLayers.size() - 1], 1)
    );
    
    /********************************************************/
    // Initialize training data 
    /********************************************************/

    // - Reinterpreting OpenFOAM's fields as torch::tensors without copying
    //  - Reinterpret OpenFOAM's input volScalarField as scalar* array 
    volScalarField::pointer vf_data = vf.ref().data();
    //  - Use the scalar* (volScalarField::pointer) to view 
    //    the volScalarField as torch::Tensor without copying data. 
    torch::Tensor vf_tensor = torch::from_blob(vf_data, {vf.size(), 1});
    //  - Reinterpret OpenFOAM's vectorField as vector* array 
    volVectorField& cc = const_cast<volVectorField&>(mesh.C()); // CELL CENTERS Spatial locations: (x,y,z)
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
    
    // - Use 10% of random indices to select the training_data from vf_tensor 
    torch::Tensor vf_training = vf_tensor.index(training_indices);
    vf_training.requires_grad_(true);
    torch::Tensor cc_training = cc_tensor.index(training_indices);
    cc_training.requires_grad_(true);
    
    /******************************************************/
    // Train the network
    /******************************************************/

    torch::optim::RMSprop optimizer(nn->parameters(), optimizerStep); // Optimiser

    torch::Tensor vf_predict = torch::zeros_like(vf_training); // Initialize the array to 0s. 
    torch::Tensor mse = torch::zeros_like(vf_training); // This would be a tensor..
    
    size_t epoch = 1; // Can be inceased further
    double min_mse = 1.; // Can be set to an even lower value


    // - Approximate DELTA_X on unstructured meshes
    const auto& deltaCoeffs = mesh.deltaCoeffs().internalField();
    double delta_x = Foam::pow(
        Foam::min(deltaCoeffs).value(),-1
    );
    
    // - Open the data file for writing the hyperparameters and logs
    auto file_name = getAvailableFileName("pinnFoam");   
    std::ofstream dataFile (file_name);
    dataFile << "HIDDEN_LAYERS,OPTIMIZER_STEP,MAX_ITERATIONS,"
        << "DELTA_X,EPOCH,DATA_MSE,GRAD_MSE,TRAINING_MSE\n";

    // - Initialize the best model (to be saved during training)
    torch::nn::Sequential nn_best;
    for (; epoch <= maxIterations; ++epoch) 
    {
        // Training
        optimizer.zero_grad();

        // Forward Propagation: Compute the prediction from the nn. 
        vf_predict = nn->forward(cc_training);

        // Back propagation: Compute the gradient of the prediction w.r.t. input. SEE THE DOCUMENTATION
        //Gradient - N_{train} x 3
        auto vf_predict_grad = torch::autograd::grad(
           {vf_predict},//N_{train} x 1
           {cc_training}, // N_{train} x 3
           {torch::ones_like(vf_training)}, // N_{train} x 1
           true
        );

        
        //printDimensions(vf_predict_grad);
        // Compute the data mse loss.
        auto mse_data = mse_loss(vf_predict, vf_training);
        
        // Compute the gradient mse loss. - Figure out how the tensors are calculated. 
        auto mse_grad = mse_loss(
            at::norm(vf_predict_grad[0], 2, -1), 
            torch::ones_like(vf_training)
        );

        // Combine the losses into a Physics Informed Neural Network.
        mse = mse_data + mse_grad; 

        // Optimize weights of the PiNN.
        mse.backward(); 
        optimizer.step();

        std::cout << "Epoch = " << epoch << "\n"
            << "Data MSE = " << mse_data.item<double>() << "\n"
            << "Grad MSE = " << mse_grad.item<double>() << "\n"
            << "Training MSE = " << mse.item<double>() << "\n";
        
        
        //std::cout << "Size = " << at::size(vf_predict_grad[0], 0) << "\n" << "Size = " << at::size(vf_predict_grad[0], 1) << "\n";
        //std::cout << "Size = " << at::size(vf_predict_grad[0], 2) << "\n" << endl; 
        
        // Write the hiddenLayers_ network structure as a string-formatted python list.
        dataFile << "\"";
        for(decltype(hiddenLayers.size()) i = 0; i < hiddenLayers.size() - 1; ++i)
            dataFile << hiddenLayers[i] << ",";
        dataFile  << hiddenLayers[hiddenLayers.size() - 1] 
            << "\"" << ",";
        // Write the rest of the data. 
        dataFile << optimizerStep << "," << maxIterations << "," 
            << delta_x << "," << epoch << "," 
            << mse_data.item<double>() << "," 
            << mse_grad.item<double>() << ","
            << mse.item<double>() << std::endl;
        
        if (mse.item<double>() < min_mse)
        {
            min_mse = mse.item<double>();
            // Save the "best" model with the minimal MSE over all epochs.
            nn_best = nn;
        }
    }
    
    /*************************************************************/
    // Evaluate the best NN. 
    /*************************************************************/

    //  - Reinterpret OpenFOAM's output volScalarField as scalar* array 
    volScalarField::pointer vf_nn_data = vf_nn.ref().data();
    //  - Use the scalar* (volScalarField::pointer) to view 
    //    the volScalarField as torch::Tensor without copying data. 
    torch::Tensor vf_nn_tensor = torch::from_blob(vf_nn_data, {vf.size()});
    //  - Evaluate the volumeScalarField vf_nn using the best NN model.
    vf_nn_tensor = nn_best->forward(cc_tensor);
    //  - FIXME: 2022-06-01, the C++ PyTorch API does not overwrite the blob object.
    //           If a Model is coded by inheritance, maybe forward(input, output) is
    //           available, that overwrites the data in vf_nn by acting on the 
    //           non-const view of the data given by vf_nn_tensor. TM.
    forAll(vf_nn, cellI)
    {
        vf_nn[cellI] = vf_nn_tensor[cellI].item<double>();
    }
    //  - Evaluate the vf_nn boundary conditions. 
    vf_nn.correctBoundaryConditions();

    /*********************************************/
    // Error calculation and output.
    /*********************************************/
    // - Data
    error_c == Foam::mag(vf - vf_nn); // Absolute error? Tensor?
    scalar error_c_l_inf = Foam::max(error_c).value();  
    scalar error_c_mean = Foam::average(error_c).value(); 
    // - Gradient  
    volVectorField vf_grad ("vf_grad", fvc::grad(vf));
    volVectorField vf_nn_grad ("vf_nn_grad", fvc::grad(vf_nn));
    volScalarField error_grad_c ("error_grad_c", Foam::mag(vf_grad - vf_nn_grad));


    Info << "max(|field - field_nn|) = " << error_c_l_inf << endl; 
    Info << "mean(|field - field_nn|) = " << error_c_mean << endl; 
    
    // Write fields
    error_c.write();
    vf_nn.write();
    vf_nn_grad.write();
    vf_grad.write(); 
    error_grad_c.write();

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
