
from typing import Tuple
from os import remove
from os.path import join, isfile, isdir
from glob import glob
from shutil import rmtree
from pandas import read_csv, DataFrame
import torch as pt
from .environment import Environment
from ..constants import TESTCASE_PATH, DEFAULT_DTYPE
from ..utils import (check_pos_int, check_pos_float, replace_line_in_file,
                     get_time_folders, get_latest_time, replace_line_latest)


pt.set_default_tensor_type(DEFAULT_DTYPE)


def _parse_forces(path: str) -> DataFrame:
    forces = read_csv(path, sep="\t", comment="#",
                      header=None, names=["t", "cd", "cl"])
    return forces


def _parse_trajectory(path: str, n_states: int) -> DataFrame:
    names = ["t", "omega", "omega_mean", "omega_log_std", "alpha", "beta", "log_p", "entropy"]
    p_names = ["p{:d}".format(i) for i in range(n_states)]
    tr = read_csv(path, sep=",", header=0, names=names+p_names)
    return tr


class RotatingCylinder2D(Environment):
    def __init__(self, r1: float = 3.0, r2: float = 0.1):
        super(RotatingCylinder2D, self).__init__(
            join(TESTCASE_PATH, "rotatingCylinder2D"), "Allrun.pre",
            "Allrun", "Allclean", 2, 100, 1
        )
        self._r1 = r1
        self._r2 = r2
        self._initialized = True
        self._start_time = 4
        self._end_time = 6
        self._control_interval = 20
        self._train = True
        self._seed = 0
        self._action_bounds = 0.05
        self._policy = "policy.pt"

    def _reward(self, cd: pt.Tensor, cl: pt.Tensor) -> pt.Tensor:
        return self._r1 - (cd + self._r2 * cl.abs())

    @property
    def start_time(self) -> float:
        return self._start_time

    @start_time.setter
    def start_time(self, value: float):
        check_pos_float(value, "start_time", with_zero=True)
        proc = True if self.initialized else False
        new = f"        startTime     {value};"
        replace_line_latest(self.path, "U", "startTime", new, proc)
        replace_line_in_file(
            join(self.path, "system", "controlDict"),
            "timeStart",
            f"        timeStart       {value};"
        )
        self._start_time = value

    @property
    def end_time(self) -> float:
        return self._end_time

    @end_time.setter
    def end_time(self, value: float):
        check_pos_float(value, "end_time", with_zero=True)
        replace_line_in_file(
            join(self.path, "system", "controlDict"),
            "endTime",
            f"endTime         {value};"
        )
        self._end_time = value

    @property
    def control_interval(self) -> int:
        return self._control_interval

    @control_interval.setter
    def control_interval(self, value: int):
        check_pos_int(value, "control_interval")
        proc = True if self.initialized else False
        new = f"        interval        {value};"
        replace_line_latest(self.path, "U", "interval", new, proc)
        replace_line_in_file(
            join(self.path, "system", "controlDict"),
            "executeInterval",
            f"        executeInterval {value};",
        )
        replace_line_in_file(
            join(self.path, "system", "controlDict"),
            "writeInterval",
            f"        writeInterval   {value};",
        )
        self._control_interval = value

    @property
    def actions_bounds(self) -> float:
        return self._action_bounds

    @actions_bounds.setter
    def action_bounds(self, value: float):
        proc = True if self.initialized else False
        new = f"        absOmegaMax     {value:2.4f};"
        replace_line_latest(self.path, "U", "absOmegaMax", new, proc)
        self._action_bounds = value

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int):
        check_pos_int(value, "seed", with_zero=True)
        proc = True if self.initialized else False
        new = f"        seed     {value};"
        replace_line_latest(self.path, "U", "seed", new, proc)
        self._seed = value

    @property
    def policy(self) -> str:
        return self._policy

    @policy.setter
    def policy(self, value: str):
        proc = True if self.initialized else False
        new = f"        policy     {value};"
        replace_line_latest(self.path, "U", "policy", new, proc)
        self._policy = value

    @property
    def train(self) -> bool:
        return self._train

    @train.setter
    def train(self, value: bool):
        proc = True if self.initialized else False
        value_cpp = "true" if value else "false"
        new = f"        train           {value_cpp};"
        replace_line_latest(self.path, "U", "train", new, proc)
        self._train = value

    @property
    def observations(self) -> Tuple[pt.Tensor]:
        force_path = join(self.path, "postProcessing", "forces", str(self._start_time), "coefficient.dat")
        forces = _parse_forces(force_path)
        tr_path = join(self.path, "trajectory.csv")
        tr = _parse_trajectory(tr_path, self._n_states)
        p_names = ["p{:d}".format(i) for i in range(self._n_states)]
        states = pt.from_numpy(tr[p_names].values)
        actions = pt.from_numpy(tr["omega"].values)
        log_p = pt.from_numpy(tr["log_p"].values)
        cd = pt.from_numpy(forces["cd"].values)
        cl = pt.from_numpy(forces["cl"].values)
        rewards = self._reward(cd, cl)
        return states, actions, rewards, log_p

    def reset(self):
        files = ["log.pimpleFoam", "finished.txt", "trajectory.csv"]
        for f in files:
            f_path = join(self.path, f)
            if isfile(f_path):
                remove(f_path)
        post = join(self.path, "postProcessing")
        if isdir(post):
            rmtree(post)
        times = get_time_folders(join(self.path, "processor0"))
        times = [t for t in times if float(t) > self.start_time]
        for p in glob(join(self.path, "processor*")):
            for t in times:
                rmtree(join(p, t))
