#!/usr/bin/python3

# For this to work, you have to have a redisAI server somewhere
# Make sure you've done: `smart build --device cpu` (or gpu)

from smartsim import Experiment

openfoam_case = "pitzDaily"

exp = Experiment("local-db", launcher="local")

db = exp.create_database(port=8000,       # database port
                         interface="lo")  # network interface to use
print(f"Creating database on  port {db.ports}")

# by default, SmartSim never blocks execution after the database is launched.
exp.start(db)

# Print some sutff because it might take a while
print('DB started...')

blockMesh_settings = exp.create_run_settings(exe="blockMesh", 
                                            exe_args=f"-case {openfoam_case}")
                        
blockMesh_model = exp.create_model(name="blockMesh", 
                                    run_settings=blockMesh_settings)
simpleFoam_settings = exp.create_run_settings(exe="simpleRedisFoam", 
                                              exe_args=f"-case {openfoam_case}")
simpleFoam_model = exp.create_model(name="simpleRedisFoam", 
                                    run_settings=simpleFoam_settings)

# launch models, analysis, training, inference sessions, etc
# that communicate with the database using the SmartRedis clients
exp.start(blockMesh_model, block=True, summary=True)
exp.start(simpleFoam_model, block=True, summary=True)

# stop the database
exp.stop(db)
