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
   - GCC > 5
   - CMake > 3

  Since these are usually system libraries we do not install them
  for the user

After installing dependencies and setting up your environment for
building SILC, as stated above, all tests can be built with the
following command

.. code-block:: bash

  make build-tests


Setup Testing Infrastructure
============================

Before running the tests, users will have to spin up a Redis
cluster instance and set the ``SSDB`` environment variable.

To spin up a Redis cluster, use the script in ``utils/create_cluster``
as follows

.. code-block:: bash

  cd /silc                       # navigate to the top level dir of silc
  conda activate env             # activate python env with SILC requirements
  source setup_env.sh            # Setup silc environment
  cd utils/create_cluster
  python local_cluster.py        # spin up Redis cluster locally
  export SSDB="127.0.0.1:6379;"  # Set database location

  # run the tests (described below)

  cd utils/create_cluster
  python local_cluster.py --stop # stop the Redis cluster


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
