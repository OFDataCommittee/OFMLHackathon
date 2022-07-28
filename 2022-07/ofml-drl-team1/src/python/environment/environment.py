"""Base class for all environments.

The base class provides a common interface for all derived environments
and implements shared functionality. New environments should be derived
from this class.
"""

from abc import ABC, abstractmethod, abstractproperty
from os.path import join
from typing import Union, Tuple
from torch import Tensor
from ..utils import check_path, check_file, check_pos_int


class Environment(ABC):
    def __init__(self, path: str, initializer_script: str, run_script: str,
                 clean_script: str, mpi_ranks: int, n_states: int,
                 n_actions: int):
        self.path = path
        self.initializer_script = initializer_script
        self.run_script = run_script
        self.clean_script = clean_script
        self.mpi_ranks = mpi_ranks
        self.n_states = n_states
        self.n_actions = n_actions
        self._initialized = False
        self._start_time = None
        self._end_time = None
        self._control_interval = None
        self._action_bounds = None
        self._seed = None
        self._policy = None
        self._train = None
        self._observations = None

    @abstractmethod
    def reset(self):
        pass

    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, value: str):
        check_path(value)
        self._path = value

    @property
    def initializer_script(self) -> str:
        return self._initializer_script

    @initializer_script.setter
    def initializer_script(self, value: str):
        check_file(join(self.path, value))
        self._initializer_script = value

    @property
    def run_script(self) -> str:
        return self._run_script

    @run_script.setter
    def run_script(self, value: str):
        check_file(join(self.path, value))
        self._run_script = value

    @property
    def clean_script(self) -> str:
        return self._clean_script

    @clean_script.setter
    def clean_script(self, value: str):
        check_file(join(self.path, value))
        self._clean_script = value

    @property
    def mpi_ranks(self) -> int:
        return self._mpi_ranks

    @mpi_ranks.setter
    def mpi_ranks(self, value: int):
        check_pos_int(value, "mpi_ranks")
        self._mpi_ranks = value

    @property
    def n_states(self) -> int:
        return self._n_states

    @n_states.setter
    def n_states(self, value: int):
        check_pos_int(value, "n_states")
        self._n_states = value

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @n_actions.setter
    def n_actions(self, value: int):
        check_pos_int(value, "n_actions")
        self._n_actions = value

    @property
    def initialized(self):
        return self._initialized

    @initialized.setter
    def initialized(self, value):
        self._initialized = True

    @abstractproperty
    def start_time(self) -> float:
        pass

    @abstractproperty
    def end_time(self) -> float:
        pass

    @abstractproperty
    def control_interval(self) -> int:
        pass

    @abstractproperty
    def actions_bounds(self) -> Union[Tensor, float]:
        pass

    @abstractproperty
    def seed(self) -> int:
        pass

    @abstractproperty
    def policy(self) -> str:
        pass

    @abstractproperty
    def train(self) -> bool:
        pass

    @abstractproperty
    def observations(self) -> Tuple[Tensor]:
        pass

    def update_control_properties(self, start_time: float, end_time: float,
                                  control_interval: float, action_bounds: Union[Tensor, float],
                                  seed: int, policy: str, train: bool):
        self.start_time = start_time
        self.end_time = end_time
        self.control_interval = control_interval
        self.actions_bounds = action_bounds
        self.seed = seed
        self.policy = policy
        self.train = train
