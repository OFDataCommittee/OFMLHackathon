#!/usr/bin/python3

##########################################################################################
#   For this script to work, SmartSim needs to have the ability to run multi-step jobs   #
#   Tracking PR: https://github.com/CrayLabs/SmartSim/pull/251                           #
##########################################################################################

from smartsim import Experiment
import os
import shutil

from smartsim.experiment import Ensemble
from smartredis import Client

import torch as pt
import torch.linalg as linalg
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

casename = "pitzDaily"

# Create an experiment
exp = Experiment(name=casename, launcher="local")

db = exp.create_database(port=8000,       # database port
                         interface="lo")  # network interface to use
print(f"Creating database on  port {db.ports}")

# by default, SmartSim never blocks execution after the database is launched.
exp.start(db)

db_address = "127.0.0.1:8000"
client = Client(address=db_address, cluster=False)

# blockMesh and solver runners
rb = exp.create_run_settings(exe="blockMesh")
rs = exp.create_run_settings(exe="simpleRedisFoam")

# Image output format
FORMAT = 'png'

# Study parameters
N_NU_FEATURES = 12
params = {
    "nu": list(np.linspace(1e-6, 1e-3, N_NU_FEATURES)),
}

ensemble = exp.create_ensemble("viscosity-study", 
                               params=params, 
                               #params_as_args=["case"],
                               run_settings=[rb, rs],
                               perm_strategy="step")

# Copy the case dir. and process tagged files
ensemble.attach_generator_files(
        to_copy=f"./input/{casename}",
        to_configure=[f"./input/{casename}/constant/transportProperties"])

for e in ensemble:
    e.register_incoming_entity(e)

# Generate parameter variation schemes
exp.generate(ensemble, overwrite=True)

# Fix paths to retain correct structure of OpenFOAM cases
for ent in ensemble.entities:
    for f in ent.files.tagged:
        bn = os.path.basename(f)
        dn = os.path.dirname(f)
        new = os.path.relpath(f, f"./input/{casename}")
        src = f"./{exp.name}/{ensemble.name}/{ent.name}/{bn}"
        dst = f"./{exp.name}/{ensemble.name}/{ent.name}/{new}"
        shutil.move(src, dst)

# Start experiments which will run blockMesh on all possible cases
exp.start(ensemble)

# For each viscosity value, we get a pressure field in the database
# ISSUE: can only interact after all cases are done

# ASSUME there are: pressure_viscosity-study_i fields in the DB
# Corresponding viscosity vals are params["nu"][i]

# Cell Center Positions:
x = pt.from_numpy(client.get_tensor(f"viscosity-study_0.x").flatten())
y = pt.from_numpy(client.get_tensor(f"viscosity-study_0.y").flatten())

## Training data
NCELLS = 12225
p_train = pt.zeros((NCELLS, N_NU_FEATURES))
for i in range(N_NU_FEATURES):
    p_train[:, i] = pt.from_numpy(client.get_tensor(f"viscosity-study_{i}.p").flatten())
    p_train[:, i] += p_train[:,i].min()
    #plt.plot(pt.linspace(0, 2*np.pi, NCELLS), p_train[:, i], label=f"{params['nu'][i]}")
#plt.legend()
#plt.show()

# Plot training data
fig, axs = plt.subplots(N_NU_FEATURES//4+(1 if N_NU_FEATURES%4>0 else 0), 4, figsize=(2*N_NU_FEATURES, 2*4))
for i in range(N_NU_FEATURES):
    axs[i//4, i%4].scatter(x, y, c=p_train[:, i], label=f"{params['nu'][i]}")
    axs[i//4, i%4].axis('scaled')
    axs[i//4, i%4].axis('off')
    axs[i//4, i%4].set_title(f"($\\nu = $ {params['nu'][i]:.3e}, min = {p_train[:, i].min():.2f}, max = {p_train[:, i].max():.2f})")
fig.suptitle(f"Training data (pressure)")
fig.savefig(f"training-data.{FORMAT}")
plt.show()

# Get the decomposition from SVD
U, s, Vt = np.linalg.svd(p_train, full_matrices=False)
x_hat = Vt.T @ np.linalg.inv(np.diag(s)) @ U.T @ p_train.numpy()

N_NU_PREDS = 3

test_nu = pt.linspace(2e-6, 9e-4, N_NU_PREDS).unsqueeze(-1).numpy()
p_pred = U @ np.diag(s) @ x_hat

# Plot predicted data
fig, axs = plt.subplots(N_NU_PREDS, 1, sharex=True)
for i in range(N_NU_PREDS):
    axs[i].scatter(x, y, c=p_pred[:, i], label=f"{test_nu[i][0]:.4e}")
    axs[i].set_title(f"($\\nu = $ {test_nu[i][0]:.3e}, min = {p_pred[:, i].min():.2f}, max = {p_pred[:, i].max():.2f})")
    axs[i].axis('scaled')
    axs[i].axis('off')
fig.suptitle(f"Predicted data (pressure)")
fig.savefig(f"predicted-data.{FORMAT}")
plt.show()

# Verify predictions

# Parameters for verification cases
params = {
    "nu" : list(test_nu.flatten())
}
ensemble = exp.create_ensemble("viscosity-verify", 
                               params=params, 
                               #params_as_args=["case"],
                               run_settings=[rb, rs],
                               perm_strategy="all_perm")
ensemble.attach_generator_files(
        to_copy=f"./input/{casename}",
        to_configure=[f"./input/{casename}/constant/transportProperties"])
for e in ensemble:
    e.register_incoming_entity(e)

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

exp.start(ensemble)

p_verify = np.zeros((NCELLS, N_NU_PREDS))
p_err = np.zeros((NCELLS, N_NU_PREDS))
fig, axs = plt.subplots(N_NU_PREDS, 1, sharex=True)
# Compute and plot relative error for scaled pressure
for i in range(N_NU_PREDS):
    p_verify[:, i] = pt.from_numpy(client.get_tensor(f"viscosity-verify_{i}.p").flatten()).numpy()
    offset = np.max(p_verify[:, i]) - np.min(p_verify[:,i])
    p_verify[:, i] /= offset
    p_pred[:, i] /= offset
    p_err[:, i] = np.abs( p_pred[:, i] - p_verify[:, i] )/ (1 + p_verify[:,i])
    axs[i].scatter(x, y, c=p_err[:, i])
    axs[i].set_title(f"($\\nu = $ {params['nu'][i]:.2e}, min_err = {p_err[:, i].min():.2f}%, max_err = {p_err[:, i].max():.2f}%, avg_err = {np.average(p_err[:, i]):.2f}%)")
    axs[i].axis('scaled')
    axs[i].axis('off')
fig.suptitle("$\\frac{|p^*_{pred} - p^*_{solver}|}{1+|p^*_{solver}|}$ (relative error in scaled pressure)")
fig.savefig(f"relative-error.{FORMAT}")
plt.show()

exp.stop(db)
