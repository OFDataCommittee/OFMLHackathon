#!/usr/bin/python3

# This defines a smartsim experiment that runs the simpleFoam solver 
# and the pitzDaily case that uses the smartSimFunctionObject. The 
# smartSimFunctionObject writes and reads OpenFOAM fields from the 
# smartredis database. 

from smartsim import Experiment

openfoam_case = "pitzDaily"

exp = Experiment("of-function-object", launcher="local")

db = exp.create_database(port=8000,       # database port
                         interface="lo")  # network interface to use
print(f"Creating database on  port {db.ports}")

blockMesh_settings = exp.create_run_settings(exe="blockMesh", 
                                             exe_args=f"-case {openfoam_case}")
                        
blockMesh_model = exp.create_model(name="blockMesh", 
                                   run_settings=blockMesh_settings)

# SmartSim doesn't blocks the database is launched per default.
exp.start(db)

simpleFoam_settings = exp.create_run_settings(exe="simpleFoam", 
                                              exe_args=f"-case {openfoam_case}")
simpleFoam_model = exp.create_model(name="simpleFoam", 
                                    run_settings=simpleFoam_settings)

# Launch models, analysis, training, inference sessions, etc
# that communicate with the database using the SmartRedis clients
exp.start(blockMesh_model, block=True, summary=True)
exp.start(simpleFoam_model, block=True, summary=True)

# stop the database
exp.stop(db)
