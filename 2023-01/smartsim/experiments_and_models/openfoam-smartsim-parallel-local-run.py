#!/usr/bin/env python

# This defines a smartsim experiment with a minimal paralel workflow for the pitzDaily case
# - generate mesh in serial, decompose the mesh, run the solver in parallel
from smartsim import Experiment

openfoam_case="pitzDaily"

# Init OpenFOAM pitzDaily Experiment and specify to launch locally
pitzDaily_experiment = Experiment(name="pitzDaily", launcher="local")

# Mesh generation
# - Settings for the blockMesh model
blockMesh_settings = pitzDaily_experiment.create_run_settings(exe="blockMesh", 
                                                           exe_args=f"-case {openfoam_case}", 
                                                           run_command=None)

# - Create the blockMesh model and attach it to pitzDaily_experiment
blockMesh_model = pitzDaily_experiment.create_model(name="blockMesh", 
                                                 run_settings=blockMesh_settings)

# Mesh decomposition
# - Settings for the decompsePar model
decomposePar_settings = pitzDaily_experiment.create_run_settings(exe="decomposePar", 
                                                              exe_args=f"-case {openfoam_case} -force", 
                                                              run_command=None)

# - Create the decomposePar model and attach it to pitzDaily_experiment
decomposePar_model = pitzDaily_experiment.create_model(name="decomposePar", 
                                                    run_settings=decomposePar_settings)

# Solver execution - PARALLEL
# - Settings for the simpleFoam model
simpleFoam_settings = pitzDaily_experiment.create_run_settings(exe="simpleFoam", 
                                                               exe_args=f"-case {openfoam_case} -parallel", 
                                                               run_command="mpirun",
                                                               run_args={"np":4})

# - Create the simpleFoam model and attach it to pitzDaily_experiment
simpleFoam_model = pitzDaily_experiment.create_model(name="simpleFoam", 
                                                     run_settings=simpleFoam_settings)


# Start the experiment
# pitzDaily_experiment.start(blockMesh_model, block=True, summary=True)
pitzDaily_experiment.start(decomposePar_model, block=True, summary=True)
pitzDaily_experiment.start(simpleFoam_model, block=True, summary=True)
