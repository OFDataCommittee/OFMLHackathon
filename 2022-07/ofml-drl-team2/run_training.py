
from src.python.buffer import LocalBuffer, SlurmBuffer
from src.python.agent import PPOAgent
from src.python.agentBayes import PPOAgentBayesian
from src.python.environment import RotatingCylinder2D
import pickle
from shutil import copytree
import os
from os import makedirs
from os.path import join
from typing import List
from torch import Tensor
import argparse

def parseArguments():

    ag = argparse.ArgumentParser()
    ag.add_argument('-d', '--directory',
                    help='Specify case directory, i.e. training path.')
    ag.add_argument('-a','--agent', choices=['PPOAgent', 'PPOAgentBayesian'], required=False, default='PPOAgent',
                    help='Select agent type, default PPOAgent')
    ag.add_argument('-n', '--nrunner', type=int, required=False, default=10,
                    help='Specify number of runners for training, default=10')
    ag.add_argument('-s', '--buffer-size', type=int, required=False, default=10,
                    help='Specify buffer-size of training, default=10')
    ag.add_argument('-e', '--episodes', type=int, required=False, default=10,
                    help='Specify number of episodes for training, default=10')
    ag.add_argument('-t', '--end-time', type=float, required=False, default=6.,
                    help='Specify end time of simulation, i.e. trajectory length, default=6.0')
    ag.add_argument('-b', '--buffer-type', type=str, choices=['local', 'slurm'], required=False, default='local',
            help='Specify buffer type to use for training: Local = Local on your machine, slurm = submit job to slurm; default=local')
    args = ag.parse_args()
    return args


def print_statistics(actions: List[Tensor], rewards: List[Tensor]):
    rt = [r.mean().item() for r in rewards]
    at_mean = [a.mean().item() for a in actions]
    at_std = [a.std().item() for a in actions]
    print("Reward mean/min/max: ", sum(rt)/len(rt), min(rt), max(rt))
    print("Mean action mean/min/max: ", sum(at_mean) /
          len(at_mean), min(at_mean), max(at_mean))
    print("Std. action mean/min/max: ", sum(at_std) /
          len(at_std), min(at_std), max(at_std))


def main(args):

    # setting
    training_path = args.directory
    episodes = args.episodes
    buffer_size = args.buffer_size
    n_runners = args.n_runners
    end_time = args.end_time

    # create a directory for training
    makedirs(training_path, exist_ok=True)

    # make a copy of the base environment
    copytree(join("test_cases", "rotatingCylinder2D"),
             join(training_path, "base"), dirs_exist_ok=True)
    env = RotatingCylinder2D()
    env.path = join(training_path, "base")
    env.end_time = end_time
    env.action_bounds = 5.
    env.reset()

    # create a trajectory buffer
    buffer = LocalBuffer(training_path, env, buffer_size, n_runners) if args.buffer_type == "local" \
                else SlurmBuffer(training_path, env, buffer_size, n_runners)

    # create PPO agent
    agent = PPOAgent(env.n_states, env.n_actions, -
                     env.action_bounds, env.action_bounds) if args.agent == "PPOAgent" else \
            PPOAgentBayesian(env.n_states, env.n_actions, -
                     env.action_bounds, env.action_bounds)

    # begin training
    for e in range(episodes):
        print(f"Start of episode {e}")
        buffer.fill()
        print("Buffer full")
        states, actions, rewards, log_p = buffer.sample()
        print_statistics(actions, rewards)
        with open(join(training_path, f"observations_e{e}.pkl"), "wb") as f:
            pickle.dump((states, actions, rewards, log_p), f, protocol=pickle.HIGHEST_PROTOCOL)
        agent.update(states, actions, rewards, log_p)
        buffer.reset()
        buffer.update_policy(agent.trace_policy())
        agent.save(join(training_path, f"policy_{e}.pt"),
                   join(training_path, f"value_{e}.pt"))
        current_policy = agent.trace_policy()
        current_policy.save(join(training_path, f"policy_trace_{e}.pt"))

    # save statistics
    with open(join(training_path, "training_history.pkl"), "wb") as f:
        pickle.dump(agent.history, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    args = parseArguments()
    main(args)
