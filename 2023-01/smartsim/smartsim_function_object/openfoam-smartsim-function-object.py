#!/usr/bin/python3

# This defines a smartsim experiment that runs the simpleFoam solver 
# and the pitzDaily case that uses the smartSimFunctionObject. The 
# smartSimFunctionObject writes and reads OpenFOAM fields from the 
# smartredis database. 
from smartsim import Experiment
from smartredis import Client 
import torch
import numpy as np

def calc_svd(input_tensor):
    # SVD function from TorchScript API
    return input_tensor.svd()

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

# TODO(TM): case parameterization 
# - Parameterize with PyFoam 
# - Ensure that fields stored in the database by the FunctionObject
#   are associated with the parameter vector. 
# - Extend the SVD to include parameterization, instead of p 
# - [p_nu1, p_nu2..]
# - [u_nu1, u_nu2..]
# - One SVD if stacked, two (n for n fields) otherwise
# - Replace p with system.controlDict.funcitons.smartRedis FO list
# - Where does the decomposition take place in parallel? 
client.set_function("svd", calc_svd)
client.run_script("svd", "calc_svd", ["p"], ["U", "S", "V"])

U = client.get_tensor("U")
S = client.get_tensor("S")
V = client.get_tensor("V")
print(f"U: {U}\n\n, S: {S}\n\n, V: {V}\n")

p_svd = np.dot(U, np.dot(S, V))
client.put_tensor("p_svd", p_svd)

# stop the database
exp.stop(db)
