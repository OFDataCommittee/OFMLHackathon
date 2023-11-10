*******************
Testing Description
*******************

##################
Quick instructions
##################

To run the tests, simply execute the following command. Omit ``SR_PYTHON=On`` or ``SR_FORTRAN=On`` if you do not wish to test those languages:
   ``make test SR_PYTHON=On SR_FORTRAN=On``

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
