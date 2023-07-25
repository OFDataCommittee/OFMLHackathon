from os import chdir, listdir, makedirs, path
from shutil import rmtree

import pytest

from gymprecice.utils.fileutils import make_env_dir, make_result_dir, open_file


FILE_CONTENT = "content"


@pytest.fixture
def testdir(tmpdir):
    test_dir = tmpdir.mkdir("test")
    yield chdir(test_dir)
    rmtree(test_dir)


def test_vaild_open_file(testdir):
    with open("open_file.txt", "w") as file:
        file.write(FILE_CONTENT)
    output = open_file("open_file.txt")
    assert output.readline() == FILE_CONTENT


def test_invalid_open_file(testdir):
    with pytest.raises(IOError):
        open_file("open_a_file_which_doesnot_exists_IOError.txt")


def test_valid_make_env_dir(testdir):
    makedirs("fluid-openfoam/content", exist_ok=True)
    with open("fluid-openfoam/content/info.txt", "w") as file:
        file.write(FILE_CONTENT)

    makedirs("solid-fenics/content", exist_ok=True)
    with open("solid-fenics/content/info.txt", "w") as file:
        file.write(FILE_CONTENT)

    makedirs("env_0", exist_ok=True)
    valid_solver_list = ["fluid-openfoam", "solid-fenics"]
    make_env_dir("env_0", valid_solver_list)

    output = {
        "soft_link_1_bool": path.islink("env_0/fluid-openfoam/content/info.txt"),
        "soft_link_2_bool": path.islink("env_0/solid-fenics/content/info.txt"),
    }
    assert all(output.values())


def test_invalid_make_env_dir(testdir):
    makedirs("fluid-openfoam/content", exist_ok=True)
    with open("fluid-openfoam/content/info.txt", "w") as file:
        file.write(FILE_CONTENT)

    makedirs("env_0", exist_ok=True)
    invalid_solver_list = ["fluid-openfoam", "no-solver-dir-provided"]
    with pytest.raises(Exception):
        make_env_dir("env_0", invalid_solver_list)


def test_valid_make_result_dir(testdir):
    makedirs("physics-simulation-engine", exist_ok=True)

    valid_environment_config = """
    {
        "environment": {
            "name": "dummy_env"
        },

        "physics_simulation_engine": {
            "solvers": ["fluid-openfoam", "solid-fenics"],
            "reset_script": "reset.sh",
            "run_script": "run.sh"
        },

        "controller": {
            "read_from": {
                "dummy_obs_interface": "dummy_obs_var"
            },
            "write_to": {
                "actuator1": "dummy_action1_variable",
                "actuator2": "dummy_action2_variable"
            }
        }
    }"""

    with open("physics-simulation-engine/gymprecice-config.json", "w") as file:
        file.write(valid_environment_config)

    xml_content = """<?xml version="1.0"?>
    ...
        <m2n:sockets from="Controller" to="Fluid" exchange-directory=""/>
        <m2n:sockets from="Controller" to="Solid" exchange-directory=""/>
    ..."""
    with open("physics-simulation-engine/precice-config.xml", "w") as file:
        file.write(xml_content)

    makedirs("physics-simulation-engine/fluid-openfoam/content", exist_ok=True)
    with open("physics-simulation-engine/fluid-openfoam/content/info.txt", "w") as file:
        file.write(FILE_CONTENT)

    makedirs("physics-simulation-engine/solid-fenics/content", exist_ok=True)
    with open("physics-simulation-engine/solid-fenics/content/info.txt", "w") as file:
        file.write(FILE_CONTENT)

    make_result_dir()

    chdir("../..")
    run_dir = path.join("gymprecice-run", listdir("gymprecice-run")[0])

    output = {
        "gymprecice-run": path.exists("gymprecice-run"),
        "precice-config.xml": path.exists(path.join(run_dir, "precice-config.xml")),
        "fluid-openfoam": path.exists(path.join(run_dir, "fluid-openfoam")),
        "solid-fenics": path.exists(path.join(run_dir, "solid-fenics")),
        "fluid-openfoam-content": path.exists(
            path.join(run_dir, "fluid-openfoam", "content", "info.txt")
        ),
        "solid-fenics-content": path.exists(
            path.join(run_dir, "solid-fenics", "content", "info.txt")
        ),
    }
    assert all(output.values())
