
Developer Makefile
------------------

SmartRedis has a Makefile that automates the build and install process.
The Makefile is shown below, and in the following sections,
the process for building and installing the SmartRedis clients from
source will be described.

.. code-block:: text

    SmartRedis Makefile help

    help                           - display this makefile's help information

    Build
    -------
    pyclient                       - Build the python client bindings
    deps                           - Make SmartRedis dependencies
    lib                            - Build SmartRedis clients into a static library
    test-deps                      - Make SmartRedis testing dependencies
    test-deps-gpu                  - Make SmartRedis GPU testing dependencies
    build-tests                    - build all tests (C, C++, Fortran)
    build-test-cpp                 - build the C++ tests
    build-test-c                   - build the C tests
    build-test-fortran             - build the Fortran tests
    build-examples                 - build all examples (serial, parallel)
    build-example-serial           - buld serial examples
    build-example-parallel         - build parallel examples (requires MPI)
    clean-deps                     - remove third-party deps
    clean                          - remove builds, pyc files, .gitignore rules
    clobber                        - clean, remove deps, builds, (be careful)

    Style
    -------
    style                          - Sort imports and format with black
    check-style                    - check code style compliance
    format                         - perform code style format
    check-format                   - check code format compliance
    sort-imports                   - apply import sort ordering
    check-sort-imports             - check imports are sorted
    check-lint                     - run static analysis checks

    Documentation
    -------
    docs                           - generate project documentation
    cov                            - generate html coverage report for Python client

    Test
    -------
    test                           - Build and run all tests (C, C++, Fortran, Python)
    test-verbose                   - Build and run all tests [verbosely]
    test-c                         - Build and run all C tests
    test-cpp                       - Build and run all C++ tests
    test-py                        - run python tests
    test-fortran                   - run fortran tests
    testpy-cov                     - run python tests with coverage

Clone and Install dependencies
-------------------------------

First, clone the SmartRedis repo:

.. code-block:: bash

    git clone https://github.com/CrayLabs/SmartRedis smartredis
    cd smartredis

SmartRedis has a base set of dependencies that are required to use the
clients.  These dependencies include Hiredis, Redis-plus-plus,
Google Protobuf, and pybind11.  The dependencies can be
downloaded, built, and installed by executing the following
command in the top-level directory of SmartRedis:

.. code-block:: bash

  make deps

To build a SmartRedis client in any language, the dependencies
downloaded above need to be found by CMake. An easy way to do
this is through environment variables. To setup your environment
for building, run the following script in the top level of the
SmartRedis directory.

.. code-block:: bash

  source setup_env.sh

Building the Python Client from Source
--------------------------------------

After the dependencies have been setup, the Python client can be
built and installed. Make sure to be using the same terminal as
the one where you installed the dependencies and sourced the
``setup_env.sh`` script.

The Python client uses Pybind11 to wrap the C++ SmartRedis client
and includes a native Python layer to make function calls simpler.
By it's design, the Python client is meant to work directly with
Numpy arrays and will return any data retrieved from a database
as a Numpy type.

.. code-block:: bash

    make pyclient
    pip install -e .[dev]
    # or if using ZSH
    pip install -e .\[dev\]

Now, when inside your virtual environment, you should be able to import
the ``Client`` from ``smartredis`` as follows

.. code-block:: python

  Python 3.7.7 (default, May  7 2020, 21:25:33)
  [GCC 7.3.0] :: Anaconda, Inc. on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> from smartredis import Client
  >>>


Building SmartRedis static library from Source
----------------------------------------------

Assuming the above steps have already been done, you are now
ready to build SmartRedis as a static library.

A static library of the SmartRedis C++, C, and Fortran clients
can be built with the command:

.. code-block:: bash

  source setup_env.sh
  make lib

