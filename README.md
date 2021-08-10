

<div align="center">
    <a href="https://github.com/CrayLabs/SmartSim"><img src="https://raw.githubusercontent.com/CrayLabs/SmartSim/master/doc/images/SmartSim_Large.png" width="90%"><img></a>
    <br />
    <br />
    <div display="inline-block">
        <a href="https://github.com/CrayLabs/SmartRedis"><b>Home</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://www.craylabs.org/docs/installation.html#smartredis"><b>Install</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://www.craylabs.org/docs/smartredis.html"><b>Documentation</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://join.slack.com/t/craylabs/shared_invite/zt-nw3ag5z5-5PS4tIXBfufu1bIvvr71UA"><b>Slack</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://github.com/CrayLabs"><b>Cray Labs</b></a>&nbsp;&nbsp;&nbsp;
    </div>
    <br />
    <br />
</div>


[![License](https://img.shields.io/github/license/CrayLabs/SmartSim)](https://github.com/CrayLabs/SmartRedis/blob/master/LICENSE.md)
![GitHub last commit](https://img.shields.io/github/last-commit/CrayLabs/SmartRedis)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/smartredis)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/CrayLabs/SmartRedis)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/smartredis)
![Language](https://img.shields.io/github/languages/top/CrayLabs/SmartRedis)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

----------
# SmartRedis

SmartRedis is a collection of Redis clients that support
RedisAI capabilities and include additional
features for high performance computing (HPC) applications.
SmartRedis provides clients in the following languages:

| Language   | Version/Standard  |
|------------|:-----------------:|
| Python     |   3.7, 3.8, 3.9   |
| C++        |   C++17           |
| C          |   C99             |
| Fortran    |   Fortran 2018    |

SmartRedis is used in the [SmartSim library](https://github.com/CrayLabs/SmartSim).
SmartSim makes it easier to use common Machine Learning (ML) libraries like
PyTorch and TensorFlow in numerical simulations at scale.  SmartRedis connects
these simulations to a Redis database or Redis database cluster for
data storage, script execution, and model evaluation.  While SmartRedis
contains features for simulation workflows on supercomputers, SmartRedis
is fully functional as a RedisAI client library and can be used without
SmartSim in any Python, C++, C, or Fortran project.

## Using SmartRedis

SmartRedis installation instructions are currently hosted as part of the
[SmartSim library installation instructions](https://www.craylabs.org/docs/installation.html#smartredis)
Additionally, detailed [API documents](https://www.craylabs.org/docs/api/smartredis_api.html) are also available as
part of the SmartSim documentation.

## Dependencies

SmartRedis utilizes the following libraries.

 - [NumPy](https://github.com/numpy/numpy)
 - [Hiredis](https://github.com/redis/hiredis) 1.0.0
 - [Redis-plus-plus](https://github.com/sewenew/redis-plus-plus)  1.2.3

## Publications

The following are public presentations or publications using SmartRedis

 - [Collaboration with NCAR - CGD Seminar](https://www.youtube.com/watch?v=2e-5j427AS0)
 - [Using Machine Learning in HPC Simulations - paper (pre-print)](https://arxiv.org/abs/2104.09355)

## Cite

Please use the following citation when referencing SmartSim, SmartRedis, or any SmartSim related work.


Partee et al., “Using Machine Learning at Scale in HPC Simulations with SmartSim:
An Application to Ocean Climate Modeling,” arXiv:2104.09355, Apr. 2021,
[Online]. Available: http://arxiv.org/abs/2104.09355.

### bibtex

    ```latex
    @misc{partee2021using,
          title={Using Machine Learning at Scale in HPC Simulations with SmartSim: An Application to Ocean Climate Modeling},
          author={Sam Partee and Matthew Ellis and Alessandro Rigazzi and Scott Bachman and Gustavo Marques and Andrew Shao and Benjamin Robbins},
          year={2021},
          eprint={2104.09355},
          archivePrefix={arXiv},
          primaryClass={cs.CE}
    }
    ```
