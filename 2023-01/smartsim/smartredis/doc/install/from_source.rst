
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

Clone SmartRedis
----------------

First, clone the SmartRedis repo:

.. code-block:: bash

    git clone https://github.com/CrayLabs/SmartRedis smartredis
    cd smartredis


Building the Python Client from Source
--------------------------------------

After cloning the repository, the Python client can be
installed from source with:

.. code-block:: bash

    pip install .

If installing SmartRedis from source for development,
it is recommended that the Python client be installed with the
``-e`` and ``[dev]``:

.. code-block:: bash

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


Building SmartRedis Static Library from Source
----------------------------------------------

Assuming the above steps have already been done, you are now
ready to build SmartRedis as a static library.

A static library of the SmartRedis C++, C, and Fortran clients
can be built with the command:

.. code-block:: bash

  make lib

The SmartRedis library will be installed in
``smartredis/install/lib/`` and the SmartRedis
header files will be installed in
``smartredis/install/include/``.
The library installation can be used to easily include SmartRedis
capabilities in C++, C, and Fortran applications.
For example, the CMake instructions below illustrate how to
compile a C or C++ application with SmartRedis.

.. code-block:: text

    project(Example)

    cmake_minimum_required(VERSION 3.13)

    set(CMAKE_CXX_STANDARD 17)

    find_library(sr_lib smartredis
                 PATHS path/to/smartredis/install/lib
                 NO_DEFAULT_PATH REQUIRED
    )

    include_directories(SYSTEM
        /usr/local/include
        path/to/smartredis/install/include
    )

    # Build executables

    add_executable(example
        example.cpp
    )
    target_link_libraries(example
        ${sr_lib}
    )

Compiling a Fortran application with the SmartRedis
library is very similar to the instructions above.
The only difference is that the Fortran SmartRedis
client source files currently need to be included
in the compilation. An example CMake file is
shown below for a Fortran application.

.. code-block:: text

    project(Example)

    cmake_minimum_required(VERSION 3.13)

    enable_language(Fortran)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_C_STANDARD 99)

    set(ftn_client_src
        path/to/smartredis/src/fortran/fortran_c_interop.F90
        path/to/smartredis/src/fortran/dataset.F90
        path/to/smartredis/src/fortran/client.F90
    )

    find_library(sr_lib smartredis
                 PATHS path/to/smartredis/install/lib
                 NO_DEFAULT_PATH REQUIRED
    )

    include_directories(SYSTEM
        /usr/local/include
        path/to/smartredis/install/include
    )

    add_executable(example
    	example.F90
	    ${ftn_client_src}
    )

    target_link_libraries(example
    	${sr_lib}
    )
