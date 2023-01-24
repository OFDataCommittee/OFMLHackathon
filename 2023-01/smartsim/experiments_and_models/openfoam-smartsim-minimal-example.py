# This defines a smartsim experiment with a minimal workflow for the cavity case
# - generate mesh with blockMesh, run the simpleFoam solver 
from smartsim import Experiment

openfoam_case="pitzDaily"

# Init OpenFOAM cavity Experiment and specify to launch locally
cavity_experiment = Experiment(name=openfoam_case, launcher="local")

# Mesh generation
# - Settings for the blockMesh model
blockMesh_settings = cavity_experiment.create_run_settings(exe="blockMesh", 
                                                           exe_args=f"-case {openfoam_case}", 
                                                           run_command=None)

# - Create the blockMesh model and attach it to cavity_experiment
blockMesh_model = cavity_experiment.create_model(name="blockMesh", run_settings=blockMesh_settings)

# Solver execution
# - Settings for the simpleFoam model
simpleFoam_settings = cavity_experiment.create_run_settings(exe="simpleFoam", 
                                                            exe_args=f"-case {openfoam_case}", 
                                                            run_command=None)

# - Create the simpleFoam model and attach it to cavity_experiment
simpleFoam_model = cavity_experiment.create_model(name="simpleFoam", run_settings=simpleFoam_settings)


# Start the experiment and make sure the models wait for each other (block=True)
cavity_experiment.start(blockMesh_model, block=True, summary=True)
cavity_experiment.start(simpleFoam_model, block=True, summary=True)
