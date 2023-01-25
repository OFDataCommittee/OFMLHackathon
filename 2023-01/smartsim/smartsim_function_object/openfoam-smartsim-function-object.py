#!/usr/bin/python3

# This defines a smartsim experiment that runs the simpleFoam solver 
# and the pitzDaily case that uses the smartSimFunctionObject. The 
# smartSimFunctionObject writes and reads OpenFOAM fields from the 
# smartredis database. 
from smartsim import Experiment
from smartredis import Client 
import torch

openfoam_case = "pitzDaily"

exp = Experiment("of-function-object", launcher="local")

db = exp.create_database(port=8000,       # database port
                         interface="lo")  # network interface to use
exp.start(db)

# blockMesh settings 
blockMesh_settings = exp.create_run_settings(exe="blockMesh", 
                                             exe_args=f"-case {openfoam_case}")
# blockMesh model             
blockMesh_model = exp.create_model(name="blockMesh", 
                                   run_settings=blockMesh_settings)

# simpleFoam settings
simpleFoam_settings = exp.create_run_settings(exe="simpleFoam", 
                                              exe_args=f"-case {openfoam_case}")
# simpleFoam model
simpleFoam_model = exp.create_model(name="simpleFoam", 
                                    run_settings=simpleFoam_settings)

# Create the mesh with blockMesh
exp.start(blockMesh_model, block=True, summary=True)

# Run simpleFoam solver in the pitzDaily case with the function 
# object that stores in the database 
exp.start(simpleFoam_model, block=True, summary=True)

# Fetch the fields from the database 
client = Client(address=db.get_address()[0], cluster=False)

# We need field metadata 
# - function object is not aware of the parameterization. 
# - how does  
# parameterization, we run the parameter variation from here. 
# On the other hand,   
vfield = client.get_tensor("p")

torch.svd(torch.from_numpy(vfield))

# stop the database
exp.stop(db)
