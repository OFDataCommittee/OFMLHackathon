# OpenFOAM-ML Hackathon Deep Reinforcement Learning Folder

## Learning Objective
By the end of the Hackathon, participants will have gained knowledge and understanding of how Reinforcement Learning (RL) agents can autonomously adjust flow parameters and optimise flow characteristics in real-time, leading to improved performance and enhanced design capabilities.

We integrate RL algorithms with OpenFOAM-based Computational Fluid Dynamics (CFD) simulations using [Gym-preCICE](https://github.com/gymprecice/gymprecice), a framework to design and develop reinforcement learning environments for single- and multi-physics active flow control (AFC).

## Challenge
During the first stage we focus on the active control of the two-dimensional Rayleigh–Benard convection problem using a single RL agent (see https://arxiv.org/pdf/2304.02370.pdf). 

Using Gym-preCICE, we aim to control the temperature of 10 actuators (boundary segments) on the hot surface (floor boundary) of the setup with the goal of minimising Nusselt number (the ratio of convective to conductive heat transfer at the floor boundary).

By the end of the hackathon we aim to create a multi-agent RL environment in which we assign one agent per each actuator (10 agents). Each agent issues a specific action (temperature) based on its local observation. 

The single-agent RL-based control case for Rayleigh–Benard convection has been already provided in the `RL_control_cases`. Please see the tutorials in the original [Gym-preCICE](https://github.com/gymprecice/gymprecice) repository for more details on the structure of RL control cases.
```
rayleigh_benard_cell
├── train_ppo_controller.py
├── environment.py
└── physics-simulation-engine
    ├── gymprecice-config.json
    ├── precice-config.json
    └── fluid-openfoam
``` 

## Installation
In case you do not have `precice` installed on your system:
- for ubuntu 22.04:
```bash
wget https://github.com/precice/precice/releases/download/v2.5.0/libprecice2_2.5.0_jammy.deb
sudo apt install ./libprecice2_2.5.0_jammy.deb
```
- for lower versions
```bash
wget https://github.com/precice/precice/releases/download/v2.5.0/libprecice2_2.5.0_focal.deb
sudo apt install ./libprecice2_2.5.0_focal.deb
```
Create and activate a `conda` virtual environment:
```bash
 conda create -n gymprecice python=3.8
 conda activate gymprecice
```
Install the provided local `gymprecice` software provided:
```bash
 cd gymprecice
 pip3 install -r requirements.txt
 pip3 install .
```
Install `pyprecice`, the python-binding for precice:
```bash
 pip3 install --user pyprecice==2.0.0.1
```

Install preCICE-OpenFOAM adapetr:
```bash
git clone https://github.com/precice/openfoam-adapter.git
cd openfoam-adapter && ./Allwmake 
```

To run the Rayleigh–Benard case, within `rayleigh_benard_cell` folder, run
```bash
python3 -u train_ppo_controller.py
```

# Troubleshooting
Error: segmentation fault:

You may need to try installing a different version of `pyprecice`. For the latest updates on the issue please see the post [here](https://github.com/precice/python-bindings/issues/182)

Error: version GLIBCXX_* not found:
```bash
cd <path to your conda env>/lib ## for example ~/anaconda3/envs/gymprecice/lib
rm libstdc++.so.6* 
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```