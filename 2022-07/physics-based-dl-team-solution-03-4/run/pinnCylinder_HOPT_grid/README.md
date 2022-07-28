# To run the case
Initialise the environment where libtorch and openfoam with alias `of2112` have been installed:
`source ../../setup_torch.sh`
`of2112`

Following above steps, run the following command to just generate the data for default parameters:
`cp -r 0_OF_orig 0 && blockMesh && pinnPotentialFOAM` 

# To run the hyper parameter optimisation
Initialise the environment where libtorch and openfoam with alias `of2112` have been installed:
`source ../../setup_torch.sh`
`of2112`
Then open jupyter notebook from command line from the parent folder where the *.ipynb files are present for the particular case. 

1. Then run the `dnnFoam-grid-search.ipynb` to carry out grid search for a range of hyperparameters
2. Then run the `dnnFoam-visualize-single-training.ipynb` to visualise the results for the optimal hyperparameter setting. 
3. Then run the `dnnFoam-visualize.ipynb` to visualise the results for te hyperparameter optimisation


