""" Example training script.
"""



import argparse
from glob import glob
from shutil import copytree, rmtree
import pickle
from os.path import join
from os import makedirs, getcwd
import sys
from os import environ
from time import time
from torch import arange, linspace
from create_dummy_policy import create_dummy_policy

BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)

from drlfoam.environment import RotatingCylinder2D
from drlfoam.agent import PPOAgent
from drlfoam.execution import LocalBuffer, SlurmBuffer, SlurmConfig


def print_statistics(actions, rewards):
    rt = [r.mean().item() for r in rewards]
    at_mean = [a.mean().item() for a in actions]
    at_std = [a.std().item() for a in actions]
    print("Reward mean/min/max: ", sum(rt)/len(rt), min(rt), max(rt))
    print("Mean action mean/min/max: ", sum(at_mean) /
          len(at_mean), min(at_mean), max(at_mean))
    print("Std. action mean/min/max: ", sum(at_std) /
          len(at_std), min(at_std), max(at_std))


def parseArguments():
    ag = argparse.ArgumentParser()
    ag.add_argument("-o", "--output", required=False, default="test_training", type=str,
                    help="Where to run the training.")
    ag.add_argument("-e", "--environment", required=False, default="local", type=str,
                    help="Use 'local' for local and 'slurm' for cluster execution.")
    ag.add_argument("-i", "--iter", required=False, default=20, type=int,
                    help="Number of training episodes.")
    ag.add_argument("-r", "--runners", required=False, default=4, type=int,
                    help="Number of runners for parallel execution.")
    ag.add_argument("-b", "--buffer", required=False, default=8, type=int,
                    help="Reply buffer size.")
    ag.add_argument("-f", "--finish", required=False, default=8.0, type=float,
                    help="End time of the simulations.")
    ag.add_argument("-t", "--timeout", required=False, default=1e15, type=int,
                    help="Maximum allowed runtime of a single simulation in seconds.")
    ag.add_argument("-v", "--velocity", required=False, default=5.0, type=float, help="Inlet velocity.")
    ag.add_argument("-c", "--checkpoint", required=False, default="", type=str,
                    help="Load training state from checkpoint file.")
    args = ag.parse_args()
    return args


def main(args):
    # settings
    training_path = args.output
    episodes = args.iter
    buffer_size = args.buffer
    n_runners = args.runners
    end_time = args.finish
    executer = args.environment
    timeout = args.timeout
    checkpoint_file = args.checkpoint
    u_target = args.velocity

    # initialize policy based on new action bounds, for now just take 10 * U_infty as min/max action
    create_dummy_policy(getcwd(), abs_action=10 * u_target)

    # create a directory for training
    makedirs(training_path, exist_ok=True)

    # make a copy of the base environment
    copytree(join(BASE_PATH, "openfoam", "test_cases", "rotatingCylinder2D"),
             join(training_path, "base"), dirs_exist_ok=True)
    env = RotatingCylinder2D(n_blocks=25, delta_t=5e-4, u_infty=1.0, omega_bounds=10 * u_target)    # start u_infty

    # max. number of blocks for highest (final) u_infty
    env.n_blocks_max = 50
    env.path = join(training_path, "base")

    # create buffer
    if executer == "local":
        buffer = LocalBuffer(training_path, env, buffer_size, n_runners, timeout=timeout)
    elif executer == "slurm":
        # Typical Slurm configs for TU Braunschweig cluster
        config = SlurmConfig(
            n_tasks=2, n_nodes=1, partition="standard", time="00:45:00",        # TODO: maybe adjust during training?
            modules=["singularity/latest", "mpi/openmpi/4.1.1/gcc"]
        )
        buffer = SlurmBuffer(training_path, env,
                             buffer_size, n_runners, config, timeout=timeout)
    else:
        raise ValueError(
            f"Unknown executer {executer}; available options are 'local' and 'slurm'.")

    # adjust mesh, time step & inlet velocity for base case if necessary
    env.adjust_setup_base()

    # create PPO agent
    agent = PPOAgent(env.n_states, env.n_actions, -
                     env.action_bounds, env.action_bounds)

    # load checkpoint if provided
    if checkpoint_file:
        print(f"Loading checkpoint from file {checkpoint_file}")
        agent.load_state(join(training_path, checkpoint_file))
        starting_episode = agent.history["episode"][-1] + 1
        buffer._n_fills = starting_episode
    else:
        starting_episode = 0
        buffer.prepare()

    buffer.base_env.start_time = buffer.base_env.end_time
    buffer.base_env.end_time = round(end_time / env.u_infty, 8)     # round corresponding to setup in controlDict
    buffer.reset()

    # inlet velocities to run: each new velocity, n_blocks for mesh refinement and increase in omega
    # TODO: delta u_unfty can greater than 1 in order to accelerate training, for now just for testing purposes
    u_values, counter = arange(env.u_infty, u_target + 1), 1
    n_blocks = [int(b.item()) for b in linspace(env.n_blocks, env.n_blocks_max, len(u_values))]
    # omega = [a.item() for a in linspace(env.action_bounds, 10 * u_target, len(u_values))]

    # begin training
    start_time = time()
    for e in range(starting_episode, episodes):
        print(f"Start of episode {e}")
        buffer.fill()
        states, actions, rewards = buffer.observations
        print_statistics(actions, rewards)
        agent.update(states, actions, rewards)
        agent.save_state(join(training_path, f"checkpoint.pt"))
        current_policy = agent.trace_policy()
        buffer.update_policy(current_policy)
        current_policy.save(join(training_path, f"policy_trace_{e}.pt"))

        # increase inlet velocity after N episodes (maybe based on grad of mean rewards of last 2 episodes better)
        if e > 0 and e % 1 == 0 and env.u_infty < u_target:
            # increment inlet velocity and refine mesh
            env.u_infty = u_values[counter].item()
            env._n_blocks = n_blocks[counter]

            # clean up, adjust time step, so that sample frequency remains const.
            for dirs in [d for d in glob(training_path + "/copy_*")]:
                rmtree(dirs)
            # TODO: instead of decreasing delta t with increasing u_infty, keep Courant number const.
            # env.action_bounds = omega[counter]
            env.adjust_control_interval(last_n_blocks=n_blocks[counter-1])

            # re-run base case with new setup, round corresponding to setup in controlDict
            buffer.prepare()
            buffer.base_env.start_time = buffer.base_env.end_time
            buffer.base_env.end_time = round(end_time / env.u_infty, 8)

            # create copies of base case
            buffer.create_copies()
            buffer.reset()

            # set new action bounds
            # agent._action_max, agent._action_min = omega[counter], -omega[counter]
            counter += 1

        else:
            buffer.reset()
    print(f"Training time (s): {time() - start_time}")


if __name__ == "__main__":
    main(parseArguments())
