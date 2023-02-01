"""Create a randomly initialized policy network.
"""
import sys
from os import environ
from typing import Union

BASE_PATH = environ.get("DRL_BASE", "")
sys.path.insert(0, BASE_PATH)
import torch as pt
from drlfoam.agent import FCPolicy


def create_dummy_policy(cwd: str, abs_action: Union[int, float, pt.Tensor]) -> None:
    # in case drlfoam is  run on HPC / using container, the path is different, so remove the last directory from it
    if cwd.endswith("examples"):
        cwd = "/".join(cwd.split("/")[:-1])
    policy = FCPolicy(12, 1, -abs_action, abs_action)
    script = pt.jit.script(policy)
    script.save(cwd + "/openfoam/test_cases/rotatingCylinder2D/policy_test.pt")


if __name__ == "__main__":
    create_dummy_policy("..", 10.0)
