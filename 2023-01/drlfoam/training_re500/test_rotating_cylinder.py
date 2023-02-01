
from os.path import join, exists
from shutil import copytree, rmtree
from pytest import raises, fixture
from ..rotating_cylinder import (RotatingCylinder2D, _parse_forces,
                                 _parse_trajectory, _parse_probes)
from ...constants import TESTDATA_PATH
from ...utils import fetch_line_from_file


@fixture()
def temp_case():
    case = "rotatingCylinder2D"
    source = join(TESTDATA_PATH, case)
    dest = join("/tmp", case)
    yield copytree(source, dest, dirs_exist_ok=True)
    rmtree(dest)


def test_parse_forces(temp_case):
    path = join(temp_case, "postProcessing", "forces", "0", "coefficient.dat")
    forces = _parse_forces(path)
    assert all(forces.columns == ["t", "cd", "cl"])
    assert len(forces) == 1


def test_parse_probes(temp_case):
    path = join(temp_case, "postProcessing", "probes", "0", "p")
    n_probes = 12
    probes = _parse_probes(path, n_probes)
    assert len(probes.columns) == n_probes + 1
    assert len(probes) == 1


def test_parse_trajectory(temp_case):
    path = join(temp_case, "trajectory.csv")
    tr = _parse_trajectory(path)
    assert all(tr.columns == ["t", "omega", "alpha", "beta"])
    assert len(tr) == 1


class TestRotatingCylinder2D(object):
    def test_common(self, temp_case):
        env = RotatingCylinder2D()
        assert True

    def test_start_time(self, temp_case):
        env = RotatingCylinder2D()
        env.initialized = True
        env.path = temp_case
        env.start_time = 2.0
        lines, _ = fetch_line_from_file(
            join(env.path, "system", "controlDict"),
            "timeStart"
        )
        assert all(["2.0" in line for line in lines])

    def test_end_time(self, temp_case):
        env = RotatingCylinder2D()
        env.initialized = True
        env.path = temp_case
        env.end_time = 8.0
        line, _ = fetch_line_from_file(
            join(env.path, "system", "controlDict"),
            "endTime "
        )
        assert "8.0" in line

    def test_control_interval(self, temp_case):
        env = RotatingCylinder2D()
        env.initialized = True
        env.path = temp_case
        env.control_interval = 0.001
        lines, _ = fetch_line_from_file(
            join(env.path, "system", "controlDict"),
            "executeInterval"
        )
        assert all(["0.001" in line for line in lines])
        lines, _ = fetch_line_from_file(
            join(env.path, "system", "controlDict"),
            "writeInterval"
        )
        assert all(["0.001" in line for line in lines])

    def test_action_bounds(self, temp_case):
        env = RotatingCylinder2D()
        env.initialized = True
        env.path = temp_case
        env.action_bounds = 10.0
        line, _ = fetch_line_from_file(
            join(env.path, "processor0", "4", "U"),
            "absOmegaMax"
        )
        assert "10.0" in line

    def test_seed(self, temp_case):
        env = RotatingCylinder2D()
        env.initialized = True
        env.path = temp_case
        env.seed = 10
        line, _ = fetch_line_from_file(
            join(env.path, "processor0", "4", "U"),
            "seed"
        )
        assert "10" in line

    def test_policy(self, temp_case):
        env = RotatingCylinder2D()
        env.initialized = True
        env.path = temp_case
        env.policy = "model.pt"
        line, _ = fetch_line_from_file(
            join(env.path, "processor0", "4", "U"),
            "policy"
        )
        assert "model.pt" in line

    def test_train(self, temp_case):
        env = RotatingCylinder2D()
        env.initialized = True
        env.path = temp_case
        env.train = False
        line, _ = fetch_line_from_file(
            join(env.path, "processor0", "4", "U"),
            "train"
        )
        assert "false" in line

    def test_reset(self, temp_case):
        env = RotatingCylinder2D()
        env.initialized = True
        env.path = temp_case
        env.start_time = 4.0
        env.reset()
        assert not exists(join(env.path, "log.pimpleFoam"))
        assert not exists(join(env.path, "trajectory.csv"))
        assert not exists(join(env.path, "postProcessing"))

    def test_observations(self, temp_case):
        env = RotatingCylinder2D()
        env.initialized = True
        env.path = temp_case
        obs = env.observations
        assert len(obs.keys()) == 7
        assert all([obs[key].shape[0] == 1 for key in obs])
