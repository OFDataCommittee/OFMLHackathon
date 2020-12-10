
************
Installation
************

SILC is utilized as a library. For C, C++, and Fortran, the clients
can be compiled directly. For Python, the clients are used just like
any other pip library. However, all clients are compatible. Python
clients can get arrays put by Fortran and vice versa.

Building SILC
=============

SILC has a makefile that automates the build process.

.. code-block:: text

  SILC Makefile help

  help                           - display this makefile's help information

  Build
  -------
  pyclient                       - Build the python client bindings
  deps                           - Make SILC dependencies
  test-deps                      - Make SILC testing dependencies
  test-deps-gpu                  - Make SILC GPU testing dependencies
  build-tests                    - build all tests (C, C++, Fortran)
  build-test-cpp                 - build the C++ tests
  build-test-c                   - build the C tests
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

  Test
  -------
  test                           - Build and run all tests (C, C++, Fortran, Python)
  test-verbose                   - Build and run all tests [verbosely]
  test-c                         - Build and run all C tests
  test-cpp                       - Build and run all C++ tests
  test-py                        - run python tests
  testpy-cov                     - run python tests with coverage

This makefile contains all the functions necessary to build and utilize SILC
in your workload.


Installing Dependencies
=======================

SILC can utilize multiple sets of dependencies depending which database
is being used, which hardware that database is running on, and which
client is being used.

For building all clients for use with Redis, use the following steps

.. code-block:: bash

  make deps

This will install Redis, Hiredis (an open source C client), Pybind11,
and protobuf. All of these dependencies are needed for building all
of the clients (C, C++, Fortran, Python)


Setting up your Environment for Building
========================================

To build a SILC client in any language, the dependencies downloaded above
need to be found by CMAKE. An easy way to do this is through environment
variables. To setup your environment for building, run the following
script in the top level of the SILC directory.

.. code-block:: bash

  source setup_env.sh

After this step, the clients will be ready to compile as a library.

Building the Python Client
==========================

The Python client uses Pybind11 to wrap the C++ SILC client and includes
a native Python layer to make function calls simpler. By it's design,
the Python client is meant to work directly with Numpy arrays and will
return any data retrieved from a database as a Numpy type.

.. note::

  The python client requires Python 3.7 or greater.

To install the Python client, follow the steps below:

.. code-block:: bash

  conda activte env # activate/create a virtual environment.
  cd silc # navigate to top level of SILC
  pip install -r requirements.txt
  make deps
  source setup_env.sh
  make pyclient

After the following script is used, the Python client module
will need to be added to your ``PYTHONPATH``. Eventually this
will be handled by a Pip install, but for now, navigate to
`silc/src/python/module` and run

.. code-block:: bash

  export PYTHONPATH=$PYTHONPATH:$(pwd)

After following the above steps, the python client should be
ready for use in any python program.

.. code-block:: python

  from silc import RAIClient
  import numpy as np

  client = RAIClient(cluster=False, Fortran=False)



*******
Testing
*******

The following will demonstate how to build and run the tests for
each of the SmartSim clients.

Building the Tests
==================

To build the tests, you first need to install the dependencies for
testing. To download SILC related testing dependencies, run
the following

.. code-block:: bash

  make test-deps
  make test-deps gpu # if you plan to run Redis/AI on GPU

.. note::

  The tests require
   - MPI to be installed
   - GCC > 5
   - CMake > 3

  Since these are usually system libraries we do not install them
  for the user

After installing dependencies and setting up your environment for
building SILC, as stated above, all tests can be built with the
following command

.. code-block:: bash

  make build-tests


Running the Tests
=================

To build and run all tests, run the following command in the top
level of the silc repository.

.. code-block:: bash

  make test

You can also run tests for individual clients as follows:

.. code-block:: bash

  make test-c       # run C tests
  make test-cpp     # run C++ test
  make test-py      # run Python tests
  make testpy-cov   # run python tests with coverage