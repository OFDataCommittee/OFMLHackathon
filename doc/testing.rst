*******
Testing
*******

The following will demonstrate how to build and run the tests for
each of the SmartSim clients.

Building the Tests
==================

Before building the tests, it is assumed that the base dependencies
for SmartRedis described in the installation instructions have already
been executed.

To build the tests, you first need to install the dependencies for
testing. To download SmartRedis related testing dependencies, run
the following:

.. code-block:: bash

  make test-deps

If you wish to run tests on GPU hardware, run the following command:

.. code-block:: bash

  make test-deps-gpu

.. note::

    The test suite is currently written to be run on CPU hardware to
    test model and script executions.  Testing on GPU hardware
    currently requires modifications to the test suite.

.. note::

  The tests require
   - GCC > 5
   - CMake > 3

  Since these are usually system libraries, we do not install them
  for the user

After installing dependencies and setting up your testing environment with
``setup_test_env.sh``, all tests can be built with the following command:

.. code-block:: bash

  ./setup_test_env.sh
  make build-tests


Starting Redis
==============

Before running the tests, users will have to spin up a Redis
cluster instance and set the ``SSDB`` environment variable.

To spin up a local Redis cluster, use the script
in ``utils/create_cluster`` as follows:

.. code-block:: bash

  cd /smartredis                 # navigate to the top level dir of smartredis
  conda activate env             # activate python env with SmartRedis requirements
  source setup_test_env.sh       # Setup smartredis environment
  cd utils/create_cluster
  python local_cluster.py        # spin up Redis cluster locally
  export SSDB="127.0.0.1:6379,127.0.0.1:6380,127.0.0.1:6381"  # Set database location

  # run the tests (described below)

  cd utils/create_cluster
  python local_cluster.py --stop # stop the Redis cluster

A similar script ``utils/create_cluster/slurm_cluster.py``
assists with launching a Redis cluster for testing on
Slurm managed machines.  This script has only been tested
on a Cray XC, and it may not be portable to all machines.

Running the Tests
=================

.. note::

    If you are running the tests in a new terminal from the
    one used to build the tests and run the Redis cluster,
    remember to load your python environment with SmartRedis
    dependencies, source the ``setup_test_env.sh`` file,
    and set the ``SSDB`` environment variable.

To build and run all tests, run the following command in the top
level of the smartredis repository.

.. code-block:: bash

  make test

You can also run tests for individual clients as follows:

.. code-block:: bash

  make test-c         # run C tests
  make test-fortran   # run Fortran tests
  make test-cpp       # run all C++ tests
  make unit-test-cpp  # run unit tests for C++
  make test-py        # run Python tests
  make testpy-cov     # run python tests with coverage
  make testcpp-cpv    # run cpp unit tests with coverage
