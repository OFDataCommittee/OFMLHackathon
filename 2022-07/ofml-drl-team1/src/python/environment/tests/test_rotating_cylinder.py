
from os.path import join, exists
from shutil import copytree, rmtree
from pytest import raises, fixture
from ..rotating_cylinder import (RotatingCylinder2D, _parse_forces,
                                 _parse_trajectory)
from ...constants import TESTCASE_PATH
from ...utils import fetch_line_from_file


@fixture()
def temp_case():
    case = "rotatingCylinder2D"
    source = join(TESTCASE_PATH, case)
    dest = join("/tmp", case)
    yield copytree(source, dest, dirs_exist_ok=True)
    rmtree(dest)


def test_parse_forces(temp_case):
    path = join(temp_case, "postProcessing", "forces", "0", "coefficient.dat")
    forces = _parse_forces(path)
    assert all(forces.columns == ["t", "cd", "cl"])

def test_parse_trajectory(temp_case):
    path = join(temp_case, "trajectory.csv")
    tr = _parse_trajectory(path, 100)
    assert len(tr.columns) == 108


class TestRotatingCylinder2D(object):
    def test_common(self, temp_case):
        env = RotatingCylinder2D()
        assert True

    def test_start_time(self, temp_case):
        env = RotatingCylinder2D()
        env.path = temp_case
        env.start_time = 2.0
        line = fetch_line_from_file(
            join(env.path, "system", "controlDict"),
            "timeStart"
        )
        assert "2.0" in line
        line = fetch_line_from_file(
            join(env.path, "processor0", "4", "U"),
            "startTime"
        )
        assert "2.0" in line

    def test_end_time(self, temp_case):
        env = RotatingCylinder2D()
        env.path = temp_case
        env.end_time = 8.0
        line = fetch_line_from_file(
            join(env.path, "system", "controlDict"),
            "endTime"
        )
        assert "8.0" in line

    def test_control_interval(self, temp_case):
        env = RotatingCylinder2D()
        env.path = temp_case
        env.control_interval = 40
        line = fetch_line_from_file(
            join(env.path, "system", "controlDict"),
            "executeInterval"
        )
        assert "40" in line
        line = fetch_line_from_file(
            join(env.path, "system", "controlDict"),
            "writeInterval"
        )
        assert "40" in line
        line = fetch_line_from_file(
            join(env.path, "processor0", "4", "U"),
            "interval"
        )
        assert "40" in line

    def test_action_bounds(self, temp_case):
        env = RotatingCylinder2D()
        env.path = temp_case
        env.action_bounds = 10.0
        line = fetch_line_from_file(
            join(env.path, "processor0", "4", "U"),
            "absOmegaMax"
        )
        assert "10.0" in line

    def test_seed(self, temp_case):
        env = RotatingCylinder2D()
        env.path = temp_case
        env.seed = 10
        line = fetch_line_from_file(
            join(env.path, "processor0", "4", "U"),
            "seed"
        )
        assert "10" in line

    def test_policy(self, temp_case):
        env = RotatingCylinder2D()
        env.path = temp_case
        env.policy = "model.pt"
        line = fetch_line_from_file(
            join(env.path, "processor0", "4", "U"),
            "policy"
        )
        assert "model.pt" in line

    def test_train(self, temp_case):
        env = RotatingCylinder2D()
        env.path = temp_case
        env.train = False
        line = fetch_line_from_file(
            join(env.path, "processor0", "4", "U"),
            "train"
        )
        assert "false" in line

    def test_reset(self, temp_case):
        env = RotatingCylinder2D()
        env.path = temp_case
        env.start_time = 4.0
        env.reset()
        assert not exists(join(env.path, "log.pimpleFoam"))
        assert not exists(join(env.path, "trajectory.csv"))
        assert not exists(join(env.path, "postProcessing"))

    def test_observations(self, temp_case):
        env = RotatingCylinder2D()
        obs = env.observations
        assert len(obs) == 4
