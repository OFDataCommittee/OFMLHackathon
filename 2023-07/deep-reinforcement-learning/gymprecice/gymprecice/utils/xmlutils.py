"""A set of common utilities used for processing xml files, specifically precice-config.xml.

These are not intended as API functions, and will not remain stable over time.
"""
import copy
import fileinput
import logging
import sys
from typing import List, Optional, Tuple

import xmltodict


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def replace_keyword(
    file: str = "precice-config.xml",
    keyword: str = None,
    value: str = None,
    place_counter_postfix: Optional[bool] = False,
):
    """Find a keyword in xml file and replace its value.

    Args:
        file (str): path to the xml file.
        keyword (str): searched keyword.
        value (str): keyword value.
        place_counter_postfix (bool): if a replacement counter needs to be added to the value: value-0, value-1, etc.

    Note: this method has been written and tested only for 'precice-config.xml', and can only handle keyword replacement in formats:
    < ... "keyword" value="value" ... />
    < ... "keyword=value" ... />
    """
    replacement_cnt = 0
    fin = fileinput.input(file, inplace=True)
    new_line = ""
    for line in fin:
        replaced = False
        if keyword in line and not replaced:
            new_line = ""
            partitioned_line = line.partition(keyword)

            value_keyword = partitioned_line[2].partition("value")
            if value_keyword[1] == "value":
                if place_counter_postfix:
                    new_line = f'{partitioned_line[0]}{keyword} value="{value}-{replacement_cnt}" '
                else:
                    new_line = f'{partitioned_line[0]}{keyword} value="{value}" '
                after_value_keyword = value_keyword[2].split()
                for item in after_value_keyword[1:-1]:
                    new_line += f" {item} "
            else:
                if place_counter_postfix:
                    new_line = (
                        f'{partitioned_line[0]}{keyword}="{value}-{replacement_cnt}" '
                    )
                else:
                    new_line = f'{partitioned_line[0]}{keyword}="{value}" '
                after_keyword = partitioned_line[2].split()
                for item in after_keyword[1:-1]:
                    new_line += f" {item} "
            new_line = new_line + " />\n"

            line = new_line
            replaced = True
            replacement_cnt += 1
        sys.stdout.write(line)
    fin.close()


def _load_file(file: str = "precice-config.xml") -> str:
    """Open precice-config.xml and return the content."""
    content = None
    with open(file) as filehandle:
        content = filehandle.readlines()
    content = "\n".join(content)
    return content


def get_episode_end_time(file: str = "precice-config.xml") -> float:
    """Read 'max-time' keyword from precice-config.xml."""
    content = _load_file(file)
    xml_tree = xmltodict.parse(content, process_namespaces=False, dict_constructor=dict)
    solver_interface = xml_tree["precice-configuration"]["solver-interface"]

    max_time = None
    for key in solver_interface.keys():
        if key.rpartition(":")[0].lower() == "coupling-scheme":
            if isinstance(solver_interface[key], list):
                max_time = solver_interface[key][0]["max-time"]["@value"]
            else:
                max_time = solver_interface[key]["max-time"]["@value"]
            break
    assert max_time is not None, f"Can't find max-time keyword in {file}"

    return float(max_time)


def get_mesh_data(file: str = "precice-config.xml") -> Tuple[List, List, List, dict]:
    """Read mesh coupling information from precice-config.xml.

    Returns:
        Tuple[list, list, list, dict]: list and dictionary of variables need to be read and write in the coupling process
    """
    content = _load_file(file)
    xml_tree = xmltodict.parse(content, process_namespaces=False, dict_constructor=dict)

    solver_interface = xml_tree["precice-configuration"]["solver-interface"]
    mesh_list = []
    if "mesh" in solver_interface.keys():
        assert (
            type(solver_interface["mesh"]) == list
        ), "single-mesh coupling is not supported. Please define mesh in precice-config.xml for all the participants"

        for item_ in solver_interface["mesh"]:
            mesh_list.append(item_["@name"])

    scaler_variables = []
    if "data:scalar" in solver_interface.keys():
        if isinstance(solver_interface["data:scalar"], dict):
            solver_interface["data:scalar"] = [solver_interface["data:scalar"]]
        for item_ in solver_interface["data:scalar"]:
            scaler_variables.append(item_["@name"])

    vector_variables = []
    if "data:vector" in solver_interface.keys():
        if isinstance(solver_interface["data:vector"], dict):
            solver_interface["data:vector"] = [solver_interface["data:vector"]]
        for item_ in solver_interface["data:vector"]:
            vector_variables.append(item_["@name"])

    controller = {}
    # we only have one controller within our participants
    for item_ in xml_tree["precice-configuration"]["solver-interface"]["participant"]:
        if "controller" in item_["@name"].lower():
            controller = copy.deepcopy(item_)
            break

    controller_dict = {}
    if "read-data" in controller.keys():
        if isinstance(controller["read-data"], dict):
            controller["read-data"] = [controller["read-data"]]
        for item_ in controller["read-data"]:
            if item_["@mesh"] not in controller_dict.keys():
                controller_dict[item_["@mesh"]] = {"read": [], "write": []}
            controller_dict[item_["@mesh"]]["read"].append(item_["@name"])

    if "write-data" in controller.keys():
        if isinstance(controller["write-data"], dict):
            controller["write-data"] = [controller["write-data"]]
        for item_ in controller["write-data"]:
            if item_["@mesh"] not in controller_dict.keys():
                controller_dict[item_["@mesh"]] = {"read": [], "write": []}
            controller_dict[item_["@mesh"]]["write"].append(item_["@name"])

    return scaler_variables, vector_variables, mesh_list, controller_dict
