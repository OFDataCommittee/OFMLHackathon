*******
Testing
*******

To build and run all tests on the local host, run the following command in the top
level of the smartredis repository:

.. code-block:: bash

  make test

.. note::

  The tests require:
   - GCC >= 5
   - CMake >= 3.13

  Since these are usually system libraries, we do not install them
  for the user.

You can also run tests for individual clients as follows:

.. code-block:: bash

  make test-c         # run C tests
  make test-fortran   # run Fortran tests. Implicitly, SR_FORTRAN=ON
  make test-cpp       # run all C++ tests
  make unit-test-cpp  # run unit tests for C++
  make test-py        # run Python tests. Implicitly, SR_PYTHON=ON
  make testpy-cov     # run python tests with coverage. Implicitly, SR_PYTHON=ON SR_BUILD=COVERAGE
  make testcpp-cpv    # run cpp unit tests with coverage. Implicitly, SR_BUILD=COVERAGE


Customizing the Tests
=====================

Several Make variables can adjust the manner in which tests are run:
   - SR_BUILD: change the way that the SmartRedis library is built. (supported: Release, Debug, Coverage; default for testing is Debug)
   - SR_FORTRAN: enable Fortran language build and testing (default is OFF)
   - SR_PYTHON: enable Python language build and testing (default is OFF)
   - SR_TEST_PORT: change the base port for Redis communication (default is 6379)
   - SR_TEST_NODES: change the number of Redis shards used for testing (default is 3)
   - SR_TEST_REDIS_MODE: change the type(s) of Redis servers used for testing. Supported is Clustered, Standalone, UDS; default is Clustered)
   - SR_TEST_REDISAI_VER: change the version of RedisAI used for testing. (Default is v1.2.3; the parameter corresponds the the RedisAI gitHub branch name for the release)
   - SR_TEST_DEVICE: change the type of device to test against. (Supported is cpu, gpu; default is cpu)
   - SR_TEST_PYTEST_FLAGS: tweak flags sent to pytest when executing tests (default is -vv -s)

These variables are all orthogonal. For example, to run tests for all languages against
a standalone Redis server, execute the following command:

.. code-block:: bash

  make test SR_FORTRAN=ON SR_PYTHON=ON SR_TEST_REDIS_MODE=Standalone

Similarly, it is possible to run the tests against each type of Redis server in sequence
(all tests against a standalone Redis server, then all tests against a Clustered server,
then all tests against a standalone server with a Unix domain socket connection) via the
following command:

.. code-block:: bash

  make test SR_FORTRAN=ON SR_PYTHON=ON SR_TEST_REDIS_MODE=All

.. note::

  Unix domain socket connections are not supported on MacOS. If the SmartRedis test
  system detects that it is running on MacOS, it will automatically skip UDS testing.

