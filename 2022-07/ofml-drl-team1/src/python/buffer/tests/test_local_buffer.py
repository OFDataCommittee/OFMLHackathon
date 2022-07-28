
from os import makedirs, remove
from os.path import join, isdir, isfile
from shutil import copytree, rmtree
from pytest import fixture
import torch as pt
from ..local_buffer import LocalBuffer
from ...environment import RotatingCylinder2D
from ...agent import FCPolicy
from ...constants import TESTCASE_PATH


@fixture()
def temp_training():
    training = join("/tmp", "test_training")
    makedirs(training, exist_ok=True)
    case = "rotatingCylinder2D"
    source = join(TESTCASE_PATH, case)
    dest = join(training, case)
    copytree(source, dest, dirs_exist_ok=True)
    env = RotatingCylinder2D()
    env.path = dest
    yield (training, env)
    rmtree(training)

class TestLocalBuffer():
    def test_create_copies(self, temp_training):
        path, env = temp_training
        buffer = LocalBuffer(path, env, 2, 2)
        assert isdir(join(path, "runner_0"))
        assert isdir(join(path, "runner_0"))
        assert buffer._envs[0].path == join(path, "runner_0")
        assert buffer._envs[1].path == join(path, "runner_1")

    def test_reset(self, temp_training):
        path, env = temp_training
        buffer = LocalBuffer(path, env, 2, 2)
        buffer.reset()
        assert not isfile(join(path, "runner_0", "trajectory.csv"))
        assert not isdir(join(path, "runner_0", "postProcessing"))

    def test_update_policy(self, temp_training):
        path, env = temp_training
        buffer = LocalBuffer(path, env, 2, 2)
        env_0 = buffer._envs[0]
        remove(join(env_0.path, env_0.policy))
        policy = FCPolicy(100, 1, -10, 10)
        buffer.update_policy(pt.jit.script(policy))
        assert isfile(join(env_0.path, env_0.policy))
