"""Setups the project."""
import re

from setuptools import setup


with open("gymprecice/version.py") as file:
    module_docstring = file.readline()
    full_version = file.readline()
    assert (
        re.match(r'VERSION = "\d\.\d+\.\d+"\n', full_version).group(0) == full_version
    ), f"Unexpected version: {full_version}"
    VERSION = re.search(r"\d\.\d+\.\d+", full_version).group(0)

# Uses the readme as the description on PyPI
with open("README.md") as fh:
    long_description = ""
    header_count = 0
    for line in fh:
        if line.startswith("##"):
            header_count += 1
        if header_count < 2:
            long_description += line
        else:
            break

setup(
    name="gymprecice",
    version=VERSION,
    license="MIT",
    url="https://github.com/gymprecice/gymprecice/",
    description="Gym-preCICE is a preCICE adapter that provides a Gymnasium-like API to couple reinforcement learning and physics-based solvers for active control",
    author="Mosayeb Shams (lead-developer), Ahmed. H. Elsheikh (supervisor)",
    author_email="m.shams@hw.ac.uk, a.elsheikh@hw.ac.uk",
)
