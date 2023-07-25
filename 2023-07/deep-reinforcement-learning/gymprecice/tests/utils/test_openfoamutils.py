from os import chdir

import numpy as np
import pytest

from gymprecice.utils.fileutils import open_file
from gymprecice.utils.openfoamutils import (
    get_interface_patches,
    get_patch_geometry,
    read_line,
)


@pytest.fixture()
def test_case(tmpdir):
    test_dir = tmpdir.mkdir("test")
    test_env_dir = test_dir.mkdir("test_env")
    test_openfoam_dir = test_env_dir.mkdir("test-openfoam")
    test_openfoam_constant_dir = test_openfoam_dir.mkdir("constant")
    test_openfoam_polymesh_dir = test_openfoam_constant_dir.mkdir("polyMesh")

    boundary = """4
(
    inlet
    {
        type            patch;
        nFaces          2;
        startFace       1;
    }
    outlet
    {
        type            patch;
        nFaces          2;
        startFace       3;
    }
    wall
    {
        type            wall;
        inGroups        1(wall);
        nFaces          2;
        startFace       5;
    }
    frontBack
    {
        type            empty;
        inGroups        1(empty);
        nFaces          4;
        startFace       7;
    }
)"""
    test_openfoam_polymesh_dir.join("boundary").write(boundary)

    faces = """11
(
4(2 8 9 3)
4(0 6 8 2)
4(2 8 10 4)
4(1 3 9 7)
4(3 5 11 9)
4(4 10 11 5)
4(0 1 7 6)
4(0 2 3 1)
4(2 4 5 3)
4(6 7 9 8)
4(8 9 11 10)
)"""
    test_openfoam_polymesh_dir.join("faces").write(faces)

    points = """12
(
(0 0 -0.5)
(1 0 -0.5)
(0 0.2 -0.5)
(1 0.2 -0.5)
(0 1 -0.5)
(1 1 -0.5)
(0 0 0.5)
(1 0 0.5)
(0 0.2 0.5)
(1 0.2 0.5)
(0 1 0.5)
(1 1 0.5)
)"""
    test_openfoam_polymesh_dir.join("points").write(points)

    return test_openfoam_dir


def test_get_patch_geometry(test_case):
    chdir(test_case)
    output = get_patch_geometry(test_case, ["inlet"])
    expected = {
        "inlet": {
            "face_centre": np.array([[0.0, 0.1, 0.0], [0.0, 0.6, 0.0]]),
            "face_area_vector": np.array([[-0.2, 0.0, 0.0], [-0.8, 0.0, 0.0]]),
            "face_area_mag": np.array([0.2, 0.8]),
            "face_normal": np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        }
    }
    assert all(
        [
            np.allclose(out, expect)
            for out, expect in zip(output["inlet"].values(), expected["inlet"].values())
        ]
    )


def test_get_interface_patches(tmpdir):
    test_dir = tmpdir.mkdir("test")
    input = test_dir.join("preciceDict")
    preciceDict = """
participant Fluid;
modules (FF);
interfaces
{
  Interface1
  {
    mesh              Fluid-Mesh;
    locations         faceCenters;
    patches           (actuator1 actuator2);
    readData
    (
      Velocity
    );
    writeData
    (
    );
  };
};
"""
    input.write(preciceDict)
    output = get_interface_patches(input)
    expected = ["actuator1", "actuator2"]
    assert output == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        ("# Time  Cd  Cs  Cl", (True, None, 0, None)),
        ("0.0005  0   1   2", (False, 0.0005, 3, [0.0, 1.0, 2.0])),
    ],
)
def test_read_line(tmpdir, input, expected):
    test_dir = tmpdir.mkdir("test")
    input_file = test_dir.join("probes")
    input_file.write(input)
    file_handler = open_file(input_file)

    output = read_line(file_handler, 3)
    assert output == expected
