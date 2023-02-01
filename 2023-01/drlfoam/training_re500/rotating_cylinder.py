
from typing import Tuple, Union
from os import remove, system
from os.path import join, isfile, isdir
from glob import glob
from re import sub
from io import StringIO
from shutil import rmtree
from pandas import read_csv, DataFrame
import torch as pt
from .environment import Environment
from ..constants import TESTCASE_PATH, DEFAULT_TENSOR_TYPE
from ..utils import (check_pos_int, check_pos_float, replace_line_in_file,
                     get_time_folders, get_latest_time, replace_line_latest, fetch_line_from_file)


pt.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


def _parse_forces(path: str) -> DataFrame:
    forces = read_csv(path, sep="\t", comment="#",
                      header=None, names=["t", "cd", "cl"])
    return forces


def _parse_probes(path: str, n_probes: int) -> DataFrame:
    with open(path, "r") as pfile:
        pdata = sub("[()]", "", pfile.read())
    names = ["t"] + [f"p{i}" for i in range(n_probes)]
    return read_csv(
        StringIO(pdata), header=None, names=names, comment="#", delim_whitespace=True
    )


def _parse_trajectory(path: str) -> DataFrame:
    names = ["t", "omega", "alpha", "beta"]
    tr = read_csv(path, sep=",", header=0, names=names)
    return tr


class RotatingCylinder2D(Environment):
    def __init__(self, r1: float = 3.0, r2: float = 1, r3: float = 0.1, n_blocks: int = 25, delta_t: float = 5e-4,
                 u_infty: Union[int, float] = 1, omega_bounds: Union[int, float, pt.Tensor] = 5.0):
        super(RotatingCylinder2D, self).__init__(
            join(TESTCASE_PATH, "rotatingCylinder2D"), "Allrun.pre",
            "Allrun", "Allclean", 2, 12, 1
        )
        self._r1 = r1
        self._r2 = r2
        self._r3 = r3
        self._initialized = False
        self._start_time = 0
        self._end_time = 4
        self._control_interval = 0.01
        self._train = True
        self._seed = 0
        self._action_bounds = omega_bounds
        self._policy = "policy.pt"
        self._n_blocks = n_blocks
        self._delta_t = delta_t
        self._u_infty = u_infty
        self._initial_control_interval = 0.01

    def _reward(self, cd: pt.Tensor, cl: pt.Tensor) -> pt.Tensor:
        return self._r1 - (self._r2 * cd + self._r3 * cl.abs())

    @property
    def start_time(self) -> float:
        return self._start_time

    @start_time.setter
    def start_time(self, value: float):
        check_pos_float(value, "start_time", with_zero=True)
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
            "endTime ",
            f"endTime         {value};"
        )
        self._end_time = value

    @property
    def control_interval(self) -> int:
        return self._control_interval

    @control_interval.setter
    def control_interval(self, value: int):
        check_pos_float(value, "control_interval")
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
    def observations(self) -> dict:
        obs = {}
        try:
            times_folder_forces = glob(
                join(self.path, "postProcessing", "forces", "*"))
            force_path = join(times_folder_forces[0], "coefficient.dat")
            forces = _parse_forces(force_path)
            tr_path = join(self.path, "trajectory.csv")
            tr = _parse_trajectory(tr_path)
            times_folder_probes = glob(
                join(self.path, "postProcessing", "probes", "*"))
            probes_path = join(times_folder_probes[0], "p")
            probes = _parse_probes(probes_path, self._n_states)
            p_names = ["p{:d}".format(i) for i in range(self._n_states)]
            obs["states"] = pt.from_numpy(probes[p_names].values)
            obs["actions"] = pt.from_numpy(tr["omega"].values)
            obs["cd"] = pt.from_numpy(forces["cd"].values)
            obs["cl"] = pt.from_numpy(forces["cl"].values)
            obs["rewards"] = self._reward(obs["cd"], obs["cl"])
            obs["alpha"] = pt.from_numpy(tr["alpha"].values)
            obs["beta"] = pt.from_numpy(tr["beta"].values)
        except Exception as e:
            print("Could not parse observations: ", e)
        finally:
            return obs

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

    def adjust_mesh(self, n_blocks_old: int = 25):
        # only replace the 1st 'blocks' variable
        replace_line_in_file(join(self.path, "system", "blockMeshDict"), f"blocks {n_blocks_old};",
                             f"blocks {self._n_blocks};")

    def set_time_step(self):
        replace_line_in_file(join(self.path, "system", "controlDict"), "deltaT", f"deltaT          {self._delta_t};")

    def set_inflow_velocity(self):
        replace_line_in_file(join(self.path, "system", "setExprBoundaryFieldsDict"), "            vel",
                             f"            vel {{ dir ({self._u_infty} 0 0); }}")
        replace_line_in_file(join(self.path, "system", "controlDict"), "        magUInf",
                             f"        magUInf         {self._u_infty};")
        replace_line_in_file(join(self.path,  "0.org", "U"), "        value",
                             f"        value           uniform ({self._u_infty} 0 0);")

    def set_end_times(self):
        replace_line_in_file(join(self.path, "0.org", "U"), "        startTime",
                             f"        startTime       {self._end_time};")
        replace_line_in_file(join(self.path, "system", "controlDict"), "endTime ", f"endTime         {self._end_time};")
        replace_line_in_file(join(self.path, "system", "controlDict"), "        timeStart",
                             f"        timeStart       {self._end_time};")

        # adjust writeInterval of the base case depending on order of chosen dt, there exist multiple 'writeInterval',
        # in this case we want the 'writeInterval' for the fields, not the 'writeInterval' for the control
        possible_lines, idx = fetch_line_from_file(join(self.path, "system", "controlDict"), "writeInterval   ")
        line = [(i, l) for i, l in enumerate(possible_lines) if l.startswith("writeInterval")][0]

        # replace the line, if u_infty = 1 then use the specified dt, else round to the order of dt
        with open("/".join([self.path, "system", "controlDict"]), "r") as f:
            lines = f.readlines()
            lines[idx[line[0]]] = f"writeInterval   {10 * pow(10, round(pt.log10(pt.tensor(self._delta_t)).item(), 0))};\n"

        # and write everything back to the controlDict
        with open("/".join([self.path, "system", "controlDict"]), "w") as f_out:
            for line in lines:
                f_out.write(line)

    def set_control_interval(self):
        # provided that dt is chosen wrt to u_infty and not based on CFL, round corresponding to setup in controlDict
        self._control_interval = round(self._initial_control_interval / self._u_infty, 8)
        replace_line_in_file(join(self.path, "system", "controlDict"), "        writeInterval   ",
                             f"        writeInterval   {self._control_interval};")
        replace_line_in_file(join(self.path, "system", "controlDict"), "executeInterval",
                             f"        executeInterval {self._control_interval};")

    def adjust_setup_base(self):
        self.adjust_mesh()
        self.set_time_step()
        self.set_inflow_velocity()

        # base case runs up to t^* = 40 = t * (u / d), d = 0.1 = const., round corresponding to setup in controlDict
        self._end_time = round(4 / self._u_infty, 8)
        self.set_end_times()

        # set omega in 0.org/U
        replace_line_in_file(join(self.path, "0.org", "U"), "        absOmegaMax",
                             f"        absOmegaMax     {self._action_bounds};")

        # set sample frequency corresponding to current setup
        self.set_control_interval()

    def adjust_control_interval(self, last_n_blocks):
        # execute Allclean for base case
        self.clean_base(self.path)

        # reset end time for base case, round corresponding to setup in controlDict
        self._end_time = round(4 / self._u_infty, 8)
        self.set_end_times()

        # reduce time step based on the (default) initial one, so that sample frequency remains the same
        self._delta_t = round(5e-4 / self._u_infty, 8)
        self.set_time_step()

        # refine mesh and increase inflow velocity
        self.set_inflow_velocity()
        self.adjust_mesh(last_n_blocks)

        # set omega in 0.org/U
        replace_line_in_file(join(self.path, "0.org", "U"), "        absOmegaMax",
                             f"        absOmegaMax     {self._action_bounds};")

        # set sample frequency corresponding to current setup
        self.set_control_interval()

    def clean_base(self, path):
        system(join(path, f"./{self.clean_script}"))

    @property
    def u_infty(self):
        return self._u_infty

    @property
    def delta_t(self):
        return self._delta_t

    @u_infty.setter
    def u_infty(self, value):
        self._u_infty = value

    @property
    def n_blocks(self):
        return self._n_blocks
