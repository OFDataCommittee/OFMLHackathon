import gymnasium as gym
from gymprecice.core import Adapter
from collections import deque

from os.path import join
import numpy as np
import math
import logging

from gymprecice.utils.openfoamutils import (
    read_line,
    get_interface_patches,
    get_patch_geometry,
)
from gymprecice.utils.fileutils import open_file

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class RBCEnv(Adapter):
    def __init__(self, options, idx=0) -> None:
        super().__init__(options, idx)

        # domain geometry
        self._xmax = 6.0
        self._H = 2.0  # height of the channel (characteristic length)

        # flow/fluid properties to compute reward
        self._baseline_Nu = 4.307  # baseline Nusselt number
        self._Pr = 0.7  # Prandtl number
        self._rho = 1.18  # density
        self._nu = 0.001  # kinematic viscosity
        self._Cp = 1.0063  # heat capacity
        self._kappa = (self._nu / self._Pr) * (
            self._rho * self._Cp
        )  # thermal conductivity

        # actuators
        self._temperature_max_variation = 2.0
        self._n_actuators = 10
        self._action_interval = 30
        self._control_start_time = 0.0
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._n_actuators,),
            dtype=np.float32,
        )

        # relevant BCs
        self._floor_temperature = 301  # hot surface
        self._ceiling_temperature = 300  # cold surface

        # probes
        self._n_probes = 300
        self._n_lookback_observation = 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._n_probes * self._n_lookback_observation,),
            dtype=np.float32,
        )

        # observations and rewards are obtained from post-processed probes/observation-patches
        self._observation_info = {
            "filed_name": "T",
            "n_probes": self._n_probes,  # number of probes
            "n_lookback": self._n_lookback_observation,  # number of time lookback
            "file_path": f"/postProcessing/probes/0/T",
            "file_handler": None,
        }

        self._reward_info = {"buffer": None, "step": None}

        # find openfoam solver (we have only one openfoam solver)
        openfoam_case_name = ""
        for solver_name in self._solver_list:
            if solver_name.rpartition("-")[-1].lower() == "openfoam":
                openfoam_case_name = solver_name
        self._openfoam_solver_path = join(self._env_path, openfoam_case_name)

        openfoam_interface_patches = get_interface_patches(
            join(openfoam_case_name, "system", "preciceDict")
        )

        control_patch = []
        self.control_patch_geometric_data = {}
        for interface in self._controller_config["write_to"]:
            if interface in openfoam_interface_patches:
                control_patch.append(interface)
        self.control_patch_geometric_data = get_patch_geometry(
            openfoam_case_name, control_patch
        )
        control_patch_coords = {}
        for patch_name in self.control_patch_geometric_data.keys():
            control_patch_coords[patch_name] = [
                np.delete(coord, 2)
                for coord in self.control_patch_geometric_data[patch_name][
                    "face_centre"
                ]
            ]

        observation_patch = []
        self.observation_patch_geometric_data = {}
        for interface in self._controller_config["read_from"]:
            if interface in openfoam_interface_patches:
                observation_patch.append(interface)
        self.observation_patch_geometric_data = get_patch_geometry(
            openfoam_case_name, observation_patch
        )

        observation_patch_coords = {}
        for patch_name in self.observation_patch_geometric_data.keys():
            observation_patch_coords[patch_name] = [
                np.delete(coord, 2)
                for coord in self.observation_patch_geometric_data[patch_name][
                    "face_centre"
                ]
            ]

        patch_coords = {
            "read_from": observation_patch_coords,
            "write_to": control_patch_coords,
        }

        self._set_precice_vectices(patch_coords)

    def step(self, actions):
        return self._repeat_step(actions)

    def _get_action(self, actions, write_var_list):
        acuation_interface_field = self._action_to_patch_field(actions)
        write_data = {
            var: acuation_interface_field[var.rpartition("-")[-1]]
            for var in write_var_list
        }
        return write_data

    def _init_data(self, write_var_list):
        # initialise temperature field for the hot surface
        T_profile = {}
        for patch_name in self.control_patch_geometric_data.keys():
            Cf = self.control_patch_geometric_data[patch_name]["face_centre"]
            T_profile[patch_name] = np.full(len(Cf), self._floor_temperature)
        write_data = {var: T_profile[var.rpartition("-")[-1]] for var in write_var_list}

        return write_data

    def _get_observation(self, read_data):
        magSf = self.control_patch_geometric_data["floor"]["face_area_mag"]
        n_lookback = self._n_lookback_observation
        if self._reward_info["buffer"] is None:
            self._reward_info["buffer"] = deque(maxlen=n_lookback)
            self._reward_info["step"] = -1
        else:
            heat_flux = -1.0 * read_data["Heat-Flux"]
            self._reward_info["buffer"].append(np.dot(heat_flux, magSf))
            self._reward_info["step"] += 1
        return self._probes_to_observation()

    def _get_reward(self):
        return self._calculate_reward()

    def _close_external_resources(self):
        # close probes and forces files
        try:
            if self._observation_info["file_handler"] is not None:
                self._observation_info["file_handler"].close()
                self._observation_info["file_handler"] = None
            if self._reward_info["buffer"] is not None:
                self._reward_info["buffer"] = None
                self._reward_info["step"] = None
        except Exception as err:
            logger.error(f"Can't close probes/forces file")
            raise err

    def _repeat_step(self, actions):
        next_obs = reward = terminated = truncated = info = None
        subcycle = 0
        while subcycle < self._action_interval:
            if isinstance(actions, np.ndarray):
                next_obs, reward, terminated, truncated, info = super().step(actions)
            else:
                next_obs, reward, terminated, truncated, info = super().step(
                    actions.cpu().numpy()
                )

            subcycle += 1
            if terminated or truncated:
                self._previous_action = None
                break

        return next_obs, reward, terminated, truncated, info

    def _action_to_patch_field(self, actions):
        # temperature field of the hot surface
        T_profile = {}
        T_patch = []
        for patch_name in self.control_patch_geometric_data.keys():
            Cf = self.control_patch_geometric_data[patch_name]["face_centre"]
            x_coords = []
            for c in Cf:
                x_coords.append(c[0])
            T_patch = self._set_temperature_profile(np.array(x_coords), actions)
            T_profile[patch_name] = T_patch

        return T_profile

    def _set_temperature_profile(self, x, actions):
        values = self._temperature_max_variation * actions
        mean = values.mean()
        centered_values = values - np.array([mean] * self._n_actuators)
        normalised_values = (centered_values * self._temperature_max_variation) / max(
            1.0, np.abs(centered_values).max()
        )

        intervals = []
        temperatures = []
        for seg in range(self._n_actuators):
            x0 = seg * self._xmax / self._n_actuators
            x1 = (seg + 1) * self._xmax / self._n_actuators
            seg_temperature = self._floor_temperature + normalised_values[seg]
            intervals.append((x > x0) & (x <= x1))
            temperatures.append(seg_temperature)

        return np.piecewise(x, intervals, temperatures)

    def _probes_to_observation(self):
        self._read_probes_from_file()

        assert self._observation_info["data"], "probes-data is empty!"
        probes_data = self._observation_info["data"]
        obs = None
        if len(probes_data) < self._n_lookback_observation:
            latest_time_data = np.array([probes_data[-1][2]])
            obs = np.concatenate([latest_time_data[0]] * self._n_lookback_observation)
        else:
            latest_time_data = probes_data[-self._n_lookback_observation :]
            obs = np.concatenate([subarr[-1] for subarr in latest_time_data])
        return obs

    def _calculate_reward(self):
        magSf = self.control_patch_geometric_data["floor"]["face_area_mag"]
        if self._t > self._control_start_time:
            heat_flux_time_avg = np.mean(self._reward_info["buffer"], axis=0)
            Nu = (heat_flux_time_avg * self._H) / (
                self._kappa
                * np.sum(magSf)
                * (self._floor_temperature - self._ceiling_temperature)
            )
            reward = self._baseline_Nu - Nu
            return reward
        else:
            return 0.0

    def _read_probes_from_file(self):
        # sequential read of a single line (last line) of probes file at each RL-Gym step
        data_path = f"{self._openfoam_solver_path}{self._observation_info['file_path']}"

        logger.debug(f"reading pressure probes from: {data_path}")

        if self._observation_info["file_handler"] is None:
            file_object = open_file(data_path)
            self._observation_info["file_handler"] = file_object
            self._observation_info["data"] = []

        new_time_stamp = True
        latest_time_stamp = self._t
        if self._observation_info["data"]:
            new_time_stamp = self._observation_info["data"][-1][0] != latest_time_stamp

        if new_time_stamp:
            time_stamp = 0
            while not math.isclose(
                time_stamp, latest_time_stamp
            ):  # read till the end of a time-window
                while True:
                    is_comment, time_stamp, n_probes, probes_data = read_line(
                        self._observation_info["file_handler"],
                        self._observation_info["n_probes"],
                    )
                    if (
                        not is_comment
                        and n_probes == self._observation_info["n_probes"]
                    ):
                        break
                self._observation_info["data"].append(
                    [time_stamp, n_probes, probes_data]
                )
            assert math.isclose(
                time_stamp, latest_time_stamp
            ), f"Mismatched time data: {time_stamp} vs {self._t}"
