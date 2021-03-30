************
Installation
************

SmartRedis clients are intended to be used as a library linked into other
applications.  For C, C++, and Fortran, the clients
can be compiled as a library that is linked with an application
at compile time. For Python, the clients can be used just like
any other pip library.

SmartRedis has a Makefile that automates the build and install process.
The Makefile is shown below, and in the following sections,
the process for building and install the SmartRedis clients will
be described.

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

Installing Dependencies
=======================

SmartRedis has a base set of dependencies that are required to use the
clients.  These dependencies include Hiredis, Redis-plus-plus,
Google Protobuf, and pybind11.  The dependencies can be
downloaded, built, and installed by executing the following
command in the top-level directory of SmartRedis:

.. code-block:: bash

  make deps

Setting up your Environment for Building
========================================

To build a SmartRedis client in any language, the dependencies downloaded above
need to be found by CMake. An easy way to do this is through environment
variables. To setup your environment for building, run the following
script in the top level of the SmartRedis directory.

.. code-block:: bash

  source setup_env.sh

After this step, the clients will be ready to compile as a library.

Building SmartRedis static library
============================

A static library of the SmartRedis C++, C, and Fortran clients can be built with
the command:

.. code-block:: bash

  make lib

The SmartRedis library will be installed in ``build/libsmartredis.a``.  This library
can be used with the SmartRedis environment variables set by ``setup_env.sh``
to add SmartRedis to existing CMake builds.  For example, the CMake
instructions below illustrate how to use the environment variables
to link in the SmartRedis static library into a C++ application.

.. code-block:: text

    set(SMARTREDIS_INSTALL_PATH "path/to/your/smartredis/install/dir")

    string(CONCAT HIREDIS_LIB_PATH $ENV{HIREDIS_INSTALL_PATH} "/lib")
    find_library(HIREDIS_LIB hiredis PATHS ${HIREDIS_LIB_PATH} NO_DEFAULT_PATH REQUIRED)
    string(CONCAT HIREDIS_INCLUDE_PATH $ENV{HIREDIS_INSTALL_PATH} "/include/")

    string(CONCAT PROTOBUF_LIB_PATH $ENV{PROTOBUF_INSTALL_PATH} "/lib")
    find_library(PROTOBUF_LIB protobuf PATHS ${PROTOBUF_LIB_PATH} NO_DEFAULT_PATH REQUIRED)
    string(CONCAT PROTOBUF_INCLUDE_PATH $ENV{PROTOBUF_INSTALL_PATH} "/include/")

    string(CONCAT REDISPP_LIB_PATH $ENV{REDISPP_INSTALL_PATH} "/lib")
    find_library(REDISPP_LIB redis++ PATHS ${REDISPP_LIB_PATH} REQUIRED)
    string(CONCAT REDISPP_INCLUDE_PATH $ENV{REDISPP_INSTALL_PATH} "/include/")

    string(CONCAT SMARTREDIS_LIB_PATH ${SMARTREDIS_INSTALL_PATH} "/build")
    find_library(SMARTREDIS_LIB smartredis PATHS ${SMARTREDIS_LIB_PATH} REQUIRED)

    include_directories(${HIREDIS_INCLUDE_PATH})
    include_directories(${REDISPP_INCLUDE_PATH})
    include_directories(${PROTOBUF_INCLUDE_PATH})
    include_directories(${SMARTREDIS_INSTALL_PATH}/include)
    include_directories(${SMARTREDIS_INSTALL_PATH}/utils/protobuf)

    set(CLIENT_LIBRARIES ${REDISPP_LIB} ${HIREDIS_LIB} ${PROTOBUF_LIB} ${SMARTREDIS_LIB})

    add_executable(example
        example.cpp
    )
    target_link_libraries(example
        ${CLIENT_LIBRARIES}
    )

Building the Python Client
==========================

The Python client uses Pybind11 to wrap the C++ SmartRedis client and includes
a native Python layer to make function calls simpler. By it's design,
the Python client is meant to work directly with Numpy arrays and will
return any data retrieved from a database as a Numpy type.

.. note::

  The python client requires Python 3.7 or greater.

To install the Python client, follow the steps below:

.. code-block:: bash

  conda activate env # activate/create a virtual environment.
  cd smartredis # navigate to top level of SmartRedis
  pip install -r requirements.txt
  make deps
  source setup_env.sh
  make pyclient


After following the above steps, the python client is
ready for use in any python program.

.. code-block:: python

  from smartredis import Client
  import numpy as np

  client = Client(cluster=False, fortran=False)
