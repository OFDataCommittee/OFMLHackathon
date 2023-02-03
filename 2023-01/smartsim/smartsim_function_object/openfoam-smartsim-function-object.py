#!/usr/bin/python3

# This defines a smartsim experiment that runs the simpleFoam solver 
# and the pitzDaily case that uses the smartSimFunctionObject. The 
# smartSimFunctionObject writes and reads OpenFOAM fields from the 
# smartredis database. 

# PyFoam for OpenFOAM input file and folder manipulation 
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
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

# Parameterize the pitzDaily case for kinematic viscosity 

# - Sample the parameter space: kinematic viscosity in this example. 
transport_properties_path = os.path.join(of_case_name, 
                                         "constant/transportProperties")
# - Complex sampling of the parameter space is available in Python 
#   PyFoam parameter variation is a simple cartesian product, but 
#   its classes allow for cloning OF folders and changing input files. 
# - PARAM prefix denotes the column that stores parameter vector components
variation_df = pd.DataFrame(np.linspace(0.5e-05, 2e-05, 4),
                        columns=["PARAM_KINEMATIC_VISCOSITY"])
# - Name the index 
variation_df.index.name = "VARIATION_ID"
# - We can add more metadata if we want.
variation_df["KINEMATIC_VISCOSITY_KEY"] = "nu"
variation_df["KINEMATIC_VISCOSITY_PATH"] = "constant/transportProperties"
variation_df["VARIATION_CASE_PATH"] = "None"

# - Create OpenFOAM template case, cloned by PyFoam 
of_template_case = SolutionDirectory(of_case_name)
# - Iterate over the parameter space (sample it in complex case)

# - Fetch the list of all PARAM_ columns 
param_columns = [column for column in variation_df.columns \
                 if column.startswith("PARAM_")]

# - Create the parameter study OpenFOAM case folders. 
for id in variation_df.index: # For each parameter vector 
    case_suffix = f"{id:04d}" # Create an unique ID
    variation_case_path = of_case_name + case_suffix
    # Add path to case to the metadata
    variation_df.loc[id, "VARIATION_CASE_PATH"] = variation_case_path
    # Clone the case
    of_template_case.cloneCase(variation_case_path) 

    # Set parameters for all parameter columns
    for param_column in param_columns:
        param_value = variation_df.loc[id, param_column]
        param_path = variation_df.loc[id, param_column.lstrip("PARAM_") + "_PATH"]
        input_file_path = os.path.join(variation_case_path, 
                                       param_path) 
        print(input_file_path)
        param_key = variation_df.loc[id, param_column.lstrip("PARAM_") + "_KEY"]
        input_file_handle = ParsedParameterFile(input_file_path)
        input_file_handle[param_key] = param_value
        input_file_handle.writeFile()

# Set up the SmartSim Experiment 
#exp = Experiment("smartsim-openfoam-param-study", launcher="local")
#
#db = exp.create_database(port=8000,       # database port
#                         interface="lo")  # network interface to use
#exp.start(db)

#for id in variation_df.index: # For each parameter vector

# This is the OpenFOAM-SmartSim parameter variation loop 
#for i, nu_param in enumerate(variation_df["KINEMATIC_VISCOSITY"]):
# blockMesh settings 
#blockMesh_settings = exp.create_run_settings(exe="blockMesh", 
#                                             exe_args=f"-case {of_case_name}")
## blockMesh model             
#blockMesh_model = exp.create_model(name="blockMesh", 
#                                   run_settings=blockMesh_settings)
#
#print (blockMesh_model.run_settings)

## simpleFoam settings
#simpleFoam_settings = exp.create_run_settings(exe="simpleFoam", 
#                                              exe_args=f"-case {of_case_name}")
## simpleFoam model
#simpleFoam_model = exp.create_model(name="simpleFoam", 
#                                    run_settings=simpleFoam_settings)

## Create the mesh with blockMesh
#exp.start(blockMesh_model, block=True, summary=True)
#
## Run simpleFoam solver in the pitzDaily case with the function 
## object that stores in the database 
#exp.start(simpleFoam_model, block=True, summary=True)
#
## Fetch the fields from the database 
#client = Client(address=db.get_address()[0], cluster=False)
#
## TODO(TM): case parameterization 
## - Parameterize with PyFoam 
## - Ensure that fields stored in the database by the FunctionObject
##   are associated with the paanually (by adding created Model objects) if launching as a batch jobrolDict.funcitons.smartRedis FO list
## - Where does the decomposition take place in parallel? 
## - Add MPI_Rank to the pressure 
#client.set_function("svd", calc_svd)
#client.run_script("svd", "calc_svd", ["p"], ["U", "S", "V"])
#
#U = client.get_tensor("U")
#S = client.get_tensor("S")
#V = client.get_tensor("V")
#print(f"U: {U}\n\n, S: {S}\n\n, V: {V}\n")
#
#p_svd = np.dot(U, np.dot(S, V))
#client.put_tensor("p_svd", p_svd)
#
## stop the database
#exp.stop(db)

# - Store the parameter variation metadata and data in a CSV file.
variation_df.to_csv("parameter_variation.csv")