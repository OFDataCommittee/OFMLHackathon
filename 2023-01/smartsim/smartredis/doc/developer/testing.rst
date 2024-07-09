*******************
Testing Description
*******************

##################
Quick instructions
##################

To run the tests, assuming that all requirements have been installed
1. Activate your environment with SmartSim and SmartRedis installed
2. Modify `SR_DB_TYPE` (`Clustered` or `Standalone`) and
   `SMARTREDIS_TEST_DEVICE` (`gpu` or `cpu`) as necessary in
   `setup_test_env.sh`.
3. `source setup_test_env.sh`
4. `pushd utils/create_cluster; python local_cluster.py; popd`
5. `export SSDB="127.0.0.1:6379,127.0.0.1:6380,127.0.0.1:6381"`
5. Run the desired tests (see `make help` for more details)

###################
Unit Test Framework
###################
All unit tests for the C++ client are located at ``tests/cpp/unit-tests/`` and use the Catch2
test framework. The unit tests mostly follow a Behavior Driven Development (BDD) style by
using Catch2's ``SCENARIO``, ``GIVEN``, ``WHEN``, and ``THEN`` syntax.

Files that contain Catch2 unit tests should be prefixed with *test_* in order to keep a
consistent naming convention.

When adding new unit tests, create a new ``SCENARIO`` in the appropriate file. If no such
file exists, then it is preferred that a new file (prefixed with *test_*) is created.

In Summary
===========

    - New unit tests should be placed in ``tests/cpp/unit-tests/``
    - Testing files should be prefixed with *test_*
    - It is preferred that new unit tests are in a new ``SCENARIO``
