[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/gymprecice/gymprecice/blob/master/LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-Commit Check](https://github.com/gymprecice/gymprecice/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/gymprecice/gymprecice/actions/workflows/pre-commit.yml)
[![Build and Test](https://github.com/gymprecice/gymprecice/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/gymprecice/gymprecice/actions/workflows/build-and-test.yml)


## Gym-preCICE

Gym-preCICE is a Python [preCICE](https://precice.org/) adapter fully compliant with [Gymnasium](https://gymnasium.farama.org/) (also known as [OpenAI Gym](https://www.gymlibrary.dev/)) API to facilitate designing and developing Reinforcement Learning (RL) environments for single- and multi-physics active flow control (AFC) applications. In an actor-environment setting, Gym-preCICE takes advantage of preCICE, an open-source coupling library for partitioned multi-physics simulations, to handle information exchange between a controller (actor) and an AFC simulation environment. The developed framework results in a seamless non-invasive integration of realistic physics-based simulation toolboxes with RL algorithms.


## Installation

### Main required dependencies

**Gymnasium**:  Installed by default. Refer to [the Gymnasium](https://gymnasium.farama.org/) for more information.

**preCICE**: You need to install the preCICE library. Refer to [the preCICE documentation](https://precice.org/installation-overview.html) for information on building and installation.

**preCICE Python bindings**: Installed by default. Refer to [the python language bindings for preCICE](https://github.com/precice/python-bindings) for information.


### Installing the package

We support and test for Python versions 3.7 and higher on Linux. We recommend installing Gym-preCICE within a virtual environment, e.g. [conda](https://www.anaconda.com/products/distribution#Downloads):

- create and activate a conda virtual environment:
```bash
 conda create -n gymprecice python=3.8
 conda activate gymprecice
```


#### PIP version
- install the adapter:

```bash
python3 -m pip install gymprecice
```
- run a simple test to check `gymprecice` installation (this should pass silently without any error/warning messages):
```bash
python3 -c "import gymprecice"
```

The default installation does not include extra dependencies to run tests or tutorials. You can install these dependencies like `python3 -m pip install gymprecice[test]`, or
`python3 -m pip install gymprecice[tutorial]`, or use `python3 -m pip install gymprecice[all]` to install all extra dependencies.

#### Development version
- if you are contributing a pull request, it is best to install from the source:
```bash
git clone https://github.com/gymprecice/gymprecice.git
cd gymprecice
pip install -e .
pip install -e .[dev]
pre-commit install
```

### Testing

We use `pytest` framework to run unit tests for all modules in our package. You need to install required dependencies before running any test:
```bash
python3 -m pip install gymprecice[test]
```
- To run the full test suits:
```
pytest ./tests
```
- To run a specific unit test, e.g. to test core module (`core.py`):
```
pytest ./tests/test_core.py
```


### Usage

Please refer to [tutorials](https://github.com/gymprecice/tutorials) for the details on how to use the adapter. You can check out the [Quickstart](https://github.com/gymprecice/tutorials/tree/main/quickstart) in our [tutorials](https://github.com/gymprecice/tutorials) repository to try a ready-to-run control case. You need to install some of the required dependencies before running any tutorial:
```bash
python3 -m pip install gymprecice[tutorial]
```


## Citing Us

If you use Gym-preCICE, please cite the following paper:

```
@misc{shams2023gymprecice,
      title={Gym-preCICE: Reinforcement Learning Environments for Active Flow Control},
      author={Mosayeb Shams and Ahmed H. Elsheikh},
      year={2023},
      eprint={2305.02033},
      archivePrefix={arXiv}
}
```

## Contributing

See the contributing guidelines [CONTRIBUTING.md](https://github.com/gymprecice/gymprecice/blob/main/CONTRIBUTING.md)
for information on submitting issues and pull requests.


## The Team

Gym-preCICE and its tutorials are primarily developed and maintained by:
- Mosayeb Shams (@mosayebshams) - Lead Developer (Heriot-Watt University)
- Ahmed H. Elsheikh(@ahmed-h-elsheikh) - Co-developer and Supervisor (Heriot-Watt University)


## Acknowledgements

This work was supported by the Engineering and Physical Sciences Research Council grant number EP/V048899/1.


## License

Gym-preCICE and its tutorials are [MIT licensed](https://github.com/gymprecice/gymprecice/blob/main/LICENSE).
