#!/usr/bin/python3

from smartsim import Experiment
import os
import shutil

casename = "pitzDaily"

# Create an experiment
exp = Experiment(name=casename, launcher="local")

# blockMesh runner
rs = exp.create_run_settings(exe="blockMesh")

# Study parameters
params = {
    "n_cells": [20, 25]
}

ensemble = exp.create_ensemble("mesh-convergence-study", 
                               params=params, 
                               run_settings=rs, 
                               perm_strategy="all_perm")

# Copy the case dir. and process tagged files
ensemble.attach_generator_files(
        to_copy=f"./input/{casename}",
        to_configure=[f"./input/{casename}/system/blockMeshDict"])

# Generate parameter variation schemes
exp.generate(ensemble, overwrite=True)

# Fix paths to retain correct structure of OpenFOAM cases
for ent in ensemble.entities:
    for f in ent.files.tagged:
        bn = os.path.basename(f)
        dn = os.path.dirname(f)
        src = "./{}/{}/{}/{}".format(exp.name, ensemble.name, ent.name, bn)
        dst = "./{}/{}/{}/{}".format(exp.name, ensemble.name, ent.name,
                                     os.path.relpath(f, f"./input/{casename}"))
        shutil.move(src, dst)

# Start experiments which will run blockMesh on all possible cases
exp.start(ensemble)
