
from os.path import join
from shutil import copytree
from copy import deepcopy
import torch as pt
from .buffer import Buffer


class LocalBuffer(Buffer):
    def __init__(self, path: str, env, size: int, n_runners: int):
        self._path = path
        self._base_env = env
        self._size = size
        self._n_runners = n_runners

        self._envs = self._create_copies()
        self._states, self._actions, self._rewards, self._log_p = [], [], [], []

    def _create_copies(self):
        envs = []
        for i in range(self._n_runners):
            dest = join(self._path, f"runner_{i}")
            copytree(self._base_env.path, dest, dirs_exist_ok=True)
            envs.append(deepcopy(self._base_env))
            envs[-1].path = dest
        return envs

    def fill(self):
        # set seed
        # run case
        # fetch observations
        pass

    def sample(self):
        return self._states, self._actions, self._rewards, self._log_p

    def update_policy(self, policy):
        for env in self._envs:
            policy.save(join(env.path, env.policy))

    def reset(self):
        for env in self._envs:
            env.reset()
        self._states, self._actions, self._rewards, self._log_p = [], [], [], []

