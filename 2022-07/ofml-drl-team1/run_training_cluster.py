
from src.python.buffer import LocalBuffer, SlurmBuffer
from src.python.agent import PPOAgent
from src.python.environment import RotatingCylinder2D
import pickle
from shutil import copytree
from os import makedirs
from os.path import join
import sys
import os
sys.path.insert(0, "src")


def print_statistics(actions, rewards):
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
    training_path = f"{os.getcwd()}/test_training"
    epochs = 50
    buffer_size = 8
    n_runners = 8

    # create a directory for training
    makedirs(training_path, exist_ok=True)

    # make a copy of the base environment
    copytree(join("test_cases", "rotatingCylinder2D"),
             join(training_path, "base"), dirs_exist_ok=True)
    env = RotatingCylinder2D()
    env.path = join(training_path, "base")
    env.end_time = 10.0
    env.reset()

    # create a trajectory buffer
    buffer = SlurmBuffer(training_path, env, buffer_size, n_runners, 2)

    # create PPO agent
    agent = PPOAgent(env.n_states, env.n_actions, -
                     env.action_bounds, env.action_bounds)

    # begin training
    for e in range(epochs):
        buffer.fill_slurm()
        states, actions, rewards, log_p = buffer.sample()
        print_statistics(actions, rewards)
        with open(join(training_path, f"observations_e{e}.pkl"), "wb") as f:
            pickle.dump((states, actions, rewards, log_p), f)
        agent.update(states, actions, rewards, log_p)
        buffer.reset()
        buffer.update_policy(agent.trace_policy())
        agent.save(join(training_path, f"policy_{e}.pt"),
                   join(training_path, f"value_{e}.pt"))
        script = agent.trace_policy()
        script.save(join(training_path, f"of_policy_{e}.pt"))
    # save statistics
    with open(join(training_path, "training_history.pkl"), "wb") as f:
        pickle.dump(agent.history, f)


if __name__ == "__main__":
    main()
