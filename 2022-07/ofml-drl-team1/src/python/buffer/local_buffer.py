
from os.path import join
from shutil import copytree
from copy import deepcopy
import torch as pt
from .buffer import Buffer
from subprocess import Popen
from queue import Queue
from _thread import start_new_thread
import random

class LocalBuffer(Buffer):
    def __init__(self, path: str, env, buffer_size: int, n_runners: int):
        self._path = path
        self._base_env = env
        self._buffer_size = buffer_size
        self._n_runners = n_runners

        self._envs = self._create_copies()
        self._states, self._actions, self._rewards, self._log_p = [], [], [], []

    def _create_copies(self):
        envs = []
        for i in range(self._buffer_size):
            dest = join(self._path, f"runner_{i}")
            copytree(self._base_env.path, dest, dirs_exist_ok=True)
            envs.append(deepcopy(self._base_env))
            envs[-1].path = dest
            envs[-1].seed = random.randint(0,1E6)
        return envs

    def wait(self, proc, job_name, queue):
        try:
            proc.wait()
        finally:
            queue.put((job_name, proc.returncode))

    def fill(self):
        count_running, count_started = 0, 0
        queue = Queue()
        proc = []

        # start initial batch of simulations
        for i in range(min(self._n_runners, self._buffer_size)):
            proc.append(Popen(["./Allrun"], cwd=f"{self._path}/runner_{count_started}"))
            start_new_thread(self.wait, (proc[-1], f"job_{count_started}", queue))
            count_running += 1
            count_started += 1

        while count_running > 0:
            # wait for completion
            job_name, return_code = queue.get()
            print(f"{job_name} finished with code {return_code}")

            # start more simulations if necessary
            if self._buffer_size > count_started:
                proc.append(Popen(["./Allrun"], cwd=f"{self._path}/runner_{count_started}"))
                start_new_thread(self.wait, (proc[-1], f"job_{count_started}", queue))
                count_started += 1
                count_running += 1
            count_running -= 1


    def sample(self):
        for env in self._envs:
            s, a, r, l = env.observations
            print(f"DEBUG: len r={len(r)}")
            self._states.append(s[1:])
            self._actions.append(a[1:-1])
            self._rewards.append(r[:])
            self._log_p.append(l[1:-1])
        return self._states, self._actions, self._rewards, self._log_p

    def update_policy(self, policy):
        for env in self._envs:
            policy.save(join(env.path, env.policy))

    def reset(self):
        for env in self._envs:
            env.reset()
        self._states, self._actions, self._rewards, self._log_p = [], [], [], []

