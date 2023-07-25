from os import chdir
from shutil import rmtree

import pytest

from gymprecice.utils.xmlutils import get_episode_end_time, get_mesh_data


VALID_XML_CONTENT_0 = """<?xml version="1.0"?>
<precice-configuration>
    ...
    <solver-interface dimensions="2">
        <data:vector name="Velocity" />
        <data:scalar name="Pressure" />
        ...
        <mesh name="Fluid-Mesh">
            <use-data name="Velocity" />
            <use-data name="Displacement" />
        </mesh>
        <mesh name="Controller-Mesh">
            <use-data name="Velocity" />
            <use-data name="Pressure" />
        </mesh>
        <participant name="Fluid">
        ...
        </participant>
        <participant name="Controller">
            <use-mesh name="Controller-Mesh" provide="yes" />
            <use-mesh name="Fluid-Mesh" from="Fluid"/>
            <write-data name="Velocity" mesh="Controller-Mesh" />
            <read-data name="Pressure"  mesh="Fluid-Mesh" />
        </participant>
        ...
        <coupling-scheme:parallel-explicit>
            <max-time value="2.335" />
            <time-window-size value="0.0005" valid-digits="8" />
            ...
        </coupling-scheme:parallel-explicit>
            ...
    </solver-interface>
</precice-configuration>"""

EXPECTED_0 = {
    "scaler_list": ["Pressure"],
    "vector_list": ["Velocity"],
    "mesh_lis": ["Fluid-Mesh", "Controller-Mesh"],
    "controller_dict": {
        "Fluid-Mesh": {"read": ["Pressure"], "write": []},
        "Controller-Mesh": {"read": [], "write": ["Velocity"]},
    },
}

VALID_XML_CONTENT_1 = """<?xml version="1.0"?>
<precice-configuration>
    ...
    <solver-interface dimensions="2">
        <data:vector name="Velocity" />
        <data:scalar name="Pressure" />
        <data:scalar name="Displacement" />
        <data:vector name="Force" />
        ...
        <mesh name="Fluid-Mesh">
            <use-data name="Velocity" />
            <use-data name="Displacement" />
        </mesh>
        <mesh name="Solid-Mesh">
            <use-data name="Force" />
        </mesh>
        <mesh name="Controller-Mesh">
            <use-data name="Velocity" />
            <use-data name="Pressure" />
            <use-data name="Displacement" />
        </mesh>
        ...
        <participant name="Fluid">
        ...
        </participant>
        <participant name="Solid">
        ...
        </participant>
        <participant name="Controller">
            <use-mesh name="Controller-Mesh" provide="yes" />
            <use-mesh name="Fluid-Mesh" from="Fluid"/>
            <use-mesh name="Solid-Mesh" from="Solid"/>
            <write-data name="Velocity" mesh="Controller-Mesh" />
            <read-data name="Pressure"  mesh="Fluid-Mesh" />
            <write-data name="Force" mesh="Controller-Mesh" />
            <read-data name="Displacement" mesh="Solid-Mesh" />
        </participant>
        ...
    </solver-interface>
</precice-configuration>"""

EXPECTED_1 = {
    "scaler_list": ["Pressure", "Displacement"],
    "vector_list": ["Velocity", "Force"],
    "mesh_lis": ["Fluid-Mesh", "Solid-Mesh", "Controller-Mesh"],
    "controller_dict": {
        "Fluid-Mesh": {"read": ["Pressure"], "write": []},
        "Solid-Mesh": {"read": ["Displacement"], "write": []},
        "Controller-Mesh": {"read": [], "write": ["Velocity", "Force"]},
    },
}


@pytest.fixture
def testdir(tmpdir):
    test_dir = tmpdir.mkdir("test")
    yield chdir(test_dir)
    rmtree(test_dir)


def test_valid_get_episode_end_time(testdir):
    with open("precice-config.xml", "w") as file:
        file.write(VALID_XML_CONTENT_0)

    episode_end_time = get_episode_end_time("precice-config.xml")
    assert episode_end_time == 2.335


@pytest.mark.parametrize(
    "input, expected",
    [(VALID_XML_CONTENT_0, EXPECTED_0), (VALID_XML_CONTENT_1, EXPECTED_1)],
)
def test_valid_get_mesh_data(testdir, input, expected):
    with open("precice-config.xml", "w") as file:
        file.write(input)

    scaler_list, vector_list, mesh_list, controller_dict = get_mesh_data(
        "precice-config.xml"
    )
    output = {
        "scaler_list": scaler_list,
        "vector_list": vector_list,
        "mesh_lis": mesh_list,
        "controller_dict": controller_dict,
    }
    assert output == expected


def test_invalid_get_mesh_data(testdir):
    invalid_xml_content = """<?xml version="1.0"?>
    <precice-configuration>
        ...
        <solver-interface dimensions="2">
            ...
            <mesh name="Controller-Mesh">
                ...
            </mesh>
            ...
            <participant name="Controller">
                ...
            </participant>
            ...
        </solver-interface>
    </precice-configuration>"""
    with open("precice-config.xml", "w") as file:
        file.write(invalid_xml_content)

    with pytest.raises(AssertionError):
        get_mesh_data("precice-config.xml")
