from os import chdir
from shutil import rmtree
from typing import List

import gymnasium as gym
import numpy as np
import pytest

from tests import mocked_precice


@pytest.fixture(autouse=True)
def testdir(tmpdir):
    test_dir = tmpdir.mkdir("test")
    yield chdir(test_dir)
    rmtree(test_dir)


@pytest.fixture
def patch_adapter_helpers(mocker):
    mocker.patch("gymprecice.core.make_env_dir", return_value=None)
    mocker.patch(
        "gymprecice.core.get_mesh_data",
        return_value=(
            None,
            None,
            [],
            {"mesh_name": "dummy_mesh", "dummy_mesh": {"read": [], "write": []}},
        ),
    )
    mocker.patch("gymprecice.core.get_episode_end_time", return_value=2.0)


@pytest.fixture
def patch_subprocess(mocker):
    mocker.patch("gymprecice.core.Adapter._launch_subprocess", return_value=[])
    mocker.patch("gymprecice.core.Adapter._check_subprocess_exists", return_value=None)


@pytest.fixture(scope="class")
def mock_precice(class_mocker):
    class_mocker.patch.dict("sys.modules", {"precice": mocked_precice})
    from precice import Interface

    Interface.initialize = class_mocker.MagicMock(return_value=float(1.0))
    Interface.advance = class_mocker.MagicMock(return_value=float(1.0))
    Interface.finalize = class_mocker.MagicMock()
    Interface.get_dimensions = class_mocker.MagicMock()
    Interface.get_mesh_id = class_mocker.MagicMock()
    Interface.get_data_id = class_mocker.MagicMock()
    Interface.initialize_data = class_mocker.MagicMock()
    Interface.set_mesh_vertices = class_mocker.MagicMock()
    Interface.is_action_required = class_mocker.MagicMock(return_value=False)
    Interface.mark_action_fulfilled = class_mocker.MagicMock()
    Interface.is_coupling_ongoing = class_mocker.MagicMock(return_value=True)
    Interface.is_time_window_complete = class_mocker.MagicMock(return_value=True)
    Interface.requires_initial_data = class_mocker.MagicMock(return_value=True)
    Interface.requires_reading_checkpoint = class_mocker.MagicMock(return_value=False)
    Interface.requires_writing_checkpoint = class_mocker.MagicMock(return_value=False)
    Interface.read_block_vector_data = class_mocker.MagicMock(return_value=None)
    Interface.read_vector_data = class_mocker.MagicMock(return_value=None)
    Interface.read_block_scalar_data = class_mocker.MagicMock(return_value=None)
    Interface.read_scalar_data = class_mocker.MagicMock(return_value=None)
    Interface.write_block_vector_data = class_mocker.MagicMock(return_value=None)
    Interface.write_vector_data = class_mocker.MagicMock(return_value=None)
    Interface.write_block_scalar_data = class_mocker.MagicMock(return_value=None)
    Interface.write_scalar_data = class_mocker.MagicMock(return_value=None)


dummy_gymprecice_config = {
    "environment": {"name": "dummy_env"},
    "physics_simulation_engine": {
        "solvers": ["dummy_solver"],
        "reset_script": "dummy_reset.sh",
        "run_script": "dummy_run.sh",
    },
    "controller": {"read_from": {}, "write_to": {}},
    "precice": {"config_file": "dummy_precice_config.xml"},
}


class TestAdapter:
    def make_env(
        self,
    ):  # a wrapper to prevent 'real precice' from being added to 'sys.module'
        from gymprecice.core import Adapter

        class DummyEnv(Adapter):
            def __init__(self, options=dummy_gymprecice_config, idx=0):
                super().__init__(options, idx)
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(151, 3), dtype=np.float32
                )
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                )
                self.dummy_obs = self.observation_space.sample()
                self._set_precice_vectices({})

            def _get_action(self, *args):
                pass

            def _get_observation(self, *args):
                return self.dummy_obs

            def _get_reward(self):
                return 0.5

            def _close_external_resources(self):
                pass

            def __del__(self):
                pass

            def _init_data(self, *args) -> dict:
                pass

        return DummyEnv()

    def test_reset(
        self, testdir, patch_adapter_helpers, patch_subprocess, mock_precice
    ):
        env = self.make_env()
        output, _ = env.reset()
        assert np.array_equal(output, env.dummy_obs)

    def test_step(
        self,
        testdir,
        patch_adapter_helpers,
        patch_subprocess,
        mock_precice,
        class_mocker,
    ):
        env = self.make_env()
        env.reset()
        # step0: not terminated
        obs_step0, reward_step0, terminated_step0, truncated_step0, _ = env.step(
            env.action_space.sample()
        )
        # step1: terminated
        from precice import Interface

        Interface.is_coupling_ongoing = class_mocker.MagicMock(return_value=False)
        obs_step1, reward_step1, terminated_step1, truncated_step1, _ = env.step(
            env.action_space.sample()
        )

        check = {
            "obs_step0": np.array_equal(obs_step0, env.dummy_obs),
            "obs_step1": np.array_equal(obs_step1, env.dummy_obs),
            "reward_step0": reward_step0 == 0.5,
            "reward_step1": reward_step1 == 0.5,
            "terminated_step0": not terminated_step0,
            "terminated_step1": terminated_step1,
            "truncated_step0": not truncated_step0,
            "truncated_step1": not truncated_step1,
        }
        assert all(check.values())
