#!/usr/bin/python3

# This defines a smartsim experiment that runs the simpleFoam solver 
# and the pitzDaily case that uses the smartSimFunctionObject. The 
# smartSimFunctionObject writes a list of OpenFOAM fields to the 
# smartredis database, the ML model reads them, performs SVD and stores
# their approximate fields back into the database. 

# PyFoam for OpenFOAM input file and folder manipulation 
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
import os
import pandas as pd

from smartsim import Experiment
from smartredis import Client 
import torch
import numpy as np

def calc_svd(input_tensor):
    """
    Applies the SVD (Singular Value Decomposition) function to the input tensor
    using the TorchScript API.
    """
    return input_tensor.svd()

of_case_name = "pitzDaily"

# Set up the OpenFOAM parameter variation as a SmartSim Experiment 
exp = Experiment("smartsim-openfoam-function-object", launcher="local")

db = exp.create_database(port=8000,       # database port
                         interface="lo")  # network interface to use
exp.start(db)

# blockMesh settings 
blockMesh_settings = exp.create_run_settings(exe="blockMesh", 
                                             exe_args=f"-case {of_case_name}")
# blockMesh model             
blockMesh_model = exp.create_model(name="blockMesh", 
                                   run_settings=blockMesh_settings)

# Mesh with blockMesh
exp.start(blockMesh_model, summary=True, block=True) 
    
# simpleFoam settings
simpleFoam_settings = exp.create_run_settings(exe="simpleFoam", 
                                              exe_args=f"-case {of_case_name}")
# simpleFoam model
simpleFoam_model = exp.create_model(name="simpleFoam", 
                                    run_settings=simpleFoam_settings)

# Run simpleFoam solver
# - The pitzDaily/system/controlDict file contains the input for the function 
#   object that will within simpleFoam_model connect and write to smartredis 
exp.start(simpleFoam_model, summary=True, block=True) 


# Get the names of OpenFOAM fiels from controlDict.functionObject 
control_dict = ParsedParameterFile(os.path.join(of_case_name,
                                                "system/controlDict"))

client = Client(address=db.get_address()[0], cluster=False)
client.set_function("svd", calc_svd)

# Apply SVD to fields 
field_names = list(control_dict["functions"]['smartSim']['fieldNames'])
print(f"SVD will be performed on OpenFOAM fields {field_names}")
for field_name in field_names:

    print (f"SVD decomposition of field: {field_name}...")
    client.run_script("svd", "calc_svd", [field_name], ["U", "S", "V"])
    print ("Done.")

    U = client.get_tensor("U")
    S = client.get_tensor("S")
    V = client.get_tensor("V")

    # Compute the Singular Value Decomposition of the field
    field_svd = np.dot(U, np.dot(S, V))

    # Compute the mean error of the SVD 
    field = client.get_tensor(field_name)
    svd_rmse = np.sqrt(np.mean((field - field_svd) ** 2))
    print (f"RMSE({field_name},SVD({field_name})): {svd_rmse}")
    
exp.stop(db)