# To run the case
Initialise the environment where libtorch and openfoam with alias `of2112` have been installed:
`source ../../setup_torch.sh`
`of2112`

Following above steps, run the following command to just generate the data for default parameters:
`cp -r 0_OF_orig 0 && blockMesh && dnnPotentialFOAM` 

