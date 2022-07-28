
from src.python.buffer import LocalBuffer
from src.python.agent import PPOAgent
from src.python.environment import RotatingCylinder2D
import pickle
from shutil import copytree
import os
from os import makedirs
from os.path import join
<<<<<<< HEAD
=======
import sys
>>>>>>> origin/main
from typing import List
from torch import Tensor


<<<<<<< HEAD
=======

>>>>>>> origin/main
def print_statistics(actions: List[Tensor], rewards: List[Tensor]):
    rt = [r.mean().item() for r in rewards]
    at_mean = [a.mean().item() for a in actions]
    at_std = [a.std().item() for a in actions]
    print("Reward mean/min/max: ", sum(rt)/len(rt), min(rt), max(rt))
    print("Mean action mean/min/max: ", sum(at_mean) /
          len(at_mean), min(at_mean), max(at_mean))
    print("Std. action mean/min/max: ", sum(at_std) /
          len(at_std), min(at_std), max(at_std))


def main():

    # setting
    training_path = "test_training"
    episodes = 10
    buffer_size = 10
    n_runners = 10
    end_time = 10.0

    # create a directory for training
    makedirs(training_path, exist_ok=True)

    # make a copy of the base environment
    copytree(join("test_cases", "rotatingCylinder2D"),
             join(training_path, "base"), dirs_exist_ok=True)
    env = RotatingCylinder2D()
    env.path = join(training_path, "base")
<<<<<<< HEAD
    env.end_time = end_time
    env.action_bounds = 5
=======
    env.end_time = 6.
>>>>>>> origin/main
    env.reset()

    # create a trajectory buffer
    buffer = LocalBuffer(training_path, env, buffer_size, n_runners)

    # create PPO agent
    agent = PPOAgent(env.n_states, env.n_actions, -
                     env.action_bounds, env.action_bounds)

    # begin training
    for e in range(episodes):
        print(f"Start of episode {e}")
        buffer.fill()
        print("Buffer full")
        states, actions, rewards, log_p = buffer.sample()
        print_statistics(actions, rewards)
<<<<<<< HEAD
        with open(join(training_path, f"observations_e{e}.pkl"), "wb") as f:
            pickle.dump((states, actions, rewards, log_p), f, protocol=pickle.HIGHEST_PROTOCOL)
=======
        with open(join(os.getcwd(), training_path, f"observations_e{e}.pkl"), 'wb') as f:
            pickle.dump((states, actions, rewards, log_p), f)
>>>>>>> origin/main
        agent.update(states, actions, rewards, log_p)
        buffer.reset()
        buffer.update_policy(agent.trace_policy())
        agent.save(join(training_path, f"policy_{e}.pt"),
                   join(training_path, f"value_{e}.pt"))
        current_policy = agent.trace_policy()
        current_policy.save(join(training_path, f"policy_trace_{e}.pt"))

    # save statistics
<<<<<<< HEAD
    with open(join(training_path, "training_history.pkl"), "wb") as f:
        pickle.dump(agent.history, f, protocol=pickle.HIGHEST_PROTOCOL)
=======
    with open(join(training_path, "training_history.pkl"), 'wb') as f:
        pickle.dump(agent.history, f)
>>>>>>> origin/main


if __name__ == "__main__":
    main()
