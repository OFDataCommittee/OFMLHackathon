# SmartSim-OpenFOAM Challenge

This folder contains the solutions to the SmartSim-OpenFOAM Hackathon challenge. We have organized the solutions as examples in separate folders.

* `experiments_and_models`: minimal working examples of OpenFOAM-SmartSim usage

    - `openfoam-smartsim-minimal-example.py`: wraps OpenFOAM's mesh generation and solvere execution into SmartSim Models, packs them into a SmartSim Experiment and runs the models locally and in serial 

    - `openfoam-smartsim-parallel-local-run.py`: defines a SmartSim experiment with a minimal paralel workflow for the pitzDaily case
    
    * `smartsim_function_object`: a Function Object that writes OpenFOAM fields into the SmartRedis Database, and a Machine Learning Model (torch.svd) "trained" on OpenFOAM fields. 

    * `smartsim_function_object`: a parameter variation version of `smartsim_function_object` 
    
    * `smartredis-simpleFoam`: **EXPERIMENTAL**: a modification of the `simpleFoam` solver for communication with the smartredis database. Parameterization requires modification of smartsim.Ensemble