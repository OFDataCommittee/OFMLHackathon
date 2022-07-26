
from os.path import join
import os
from shutil import copytree
from copy import deepcopy
from subprocess import Popen
from _thread import start_new_thread
from queue import Queue
import torch as pt
from .buffer import Buffer
import numpy as np
from ..environment import Environment


class LocalBuffer(Buffer):
    def __init__(self, path: str, env: Environment, size: int, n_runners: int):
        self._path = path
        self._base_env = env
        self._size = size
        self._n_runners = n_runners

        self._envs = self._create_copies()

    def _create_copies(self):
        envs = []
        for i in range(self._size):
            dest = join(self._path, f"runner_{i}")
            copytree(self._base_env.path, dest, dirs_exist_ok=True)
            envs.append(deepcopy(self._base_env))
            envs[-1].path = dest
            envs[-1].seed = i
        return envs

    def _wait(self, proc, job_name, queue):
        try:
            proc.wait()
        finally:
            queue.put((job_name, proc.returncode))

    def fill(self):
        count_running, count_started = 0, 0
        queue = Queue()
        proc = []

        for i in range(min(self._n_runners, self._size)):
            env = self._envs[count_started]
            proc.append(
                Popen([f"./{env.run_script}"], cwd=env.path)
            )
            start_new_thread(self._wait, (proc[-1], env.path, queue))
            count_running += 1
            count_started += 1

        while count_running > 0:
            job_name, rc = queue.get()
            print(f"Finished {job_name} with return code {rc}")
            if self._size > count_started:
                env = self._envs[count_started]
                proc.append(
                    Popen([f"./{env.run_script}"], cwd=env.path)
                )
                start_new_thread(self._wait, (proc[-1], env.path, queue))
                count_started += 1
                count_running += 1
            count_running -= 1

    def sample(self):
        states, actions, rewards, log_p = [], [], [], []
        for env in self._envs:
            s, a, r, p = env.observations
            states.append(s)
            actions.append(a[:-1])
            rewards.append(r)
            log_p.append(p[:-1])
        return states, actions, rewards, log_p

    def update_policy(self, policy):
        for env in self._envs:
            policy.save(join(env.path, env.policy))

    def reset(self):
        for env in self._envs:
            env.reset()
        self._states, self._actions, self._rewards, self._log_p = [], [], [], []

    def process_waiter(self, proc, job_name, que):
        """
             This method is to wait for the executed process till it is completed
         """
        try:
            proc.wait()
        finally:
            que.put((job_name, proc.returncode))
