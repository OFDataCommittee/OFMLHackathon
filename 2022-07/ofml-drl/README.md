# DRL for flow control in OpenFOAM

This code is re-factored (and currently incomplete) version of the following repositories:

- student projects by Darshan Thummar ([link](https://github.com/darshan315/flow_past_cylinder_by_DRL)) and Fabian Gabriel ([link](https://github.com/FabianGabriel/Active_flow_control_past_cylinder_using_DRL))
- exercise 11 of [this lecture](https://github.com/AndreWeiner/ml-cfd-lecture/blob/main/notebooks/exercise_10_11.ipynb)

A brief theoretical overview may be found [here](https://andreweiner.github.io/ml-cfd-slides/lecture_10_11.html).

## Dependencies

The main dependencies are:
- OpenFOAM-v2206; standard package installation is fine
- PyTorch-1.12.0; ideally, create a virtual environment as shown below

Creating and using a virtual Python environment:
- install *venv*; on Ubuntu 20.04 and above, run `sudo apt install python3.8-venv`
- set up a new environment as show below:
```
python3 -m venv pydrl
source pydrl/bin/activate
pip install -r requirements.txt
deactivate
```

For running the unittests, you also need *pytest*.


## Compiling the C++ code

Source the environment variables and compile:

```
source setup-env
./Allwmake
```

## Running the Python code

### Unittests

```
source setup-env
source pydrl/bin/activate
pytest -s src/python
```

### Training

The main training loop is implemented in *run_training.py*. However, the `fill()` method of the `LocalBuffer` class is not yet implemented. Moreover, the `SlurmBuffer` class for execution on the cluster is missing. Once the `Buffer` is complete, the training can be started by running:

```
python3 -m run_training.py
```

## Tasks for the hackathon

1. implement the `LocalBuffer.fill()` method; [hint](https://github.com/AndreWeiner/ml-cfd-lecture/blob/main/test_cases/drl_control_cylinder/env_local.py)
2. implement the `SlurmBuffer` class analogously; [hint](https://github.com/FabianGabriel/Active_flow_control_past_cylinder_using_DRL/blob/main/DRL_py_beta/env_cluster.py)
3. implement one or more of the ideas below or your own ideas

- the main challenge in DRL with CFD is the relatively large cost of the *environment*; the training might be accelerated as follows:
  - buffer size: a larger buffer size requires more computational resources but stabilizes the training; find and implement a criterion to adjust the buffer size during training
  - trajectory length: if the trajectory is shorter, the training is accelerated, but there might be too little exploration to learn; find and implement a criterion to adjust the trajectory length during training
- one important question for applications is knowing the applicability of a policy under changed flow conditions, e.g., are we extrapolating in the feature space? Use a Bayesian network as policy to provide uncertainty bounds for the control