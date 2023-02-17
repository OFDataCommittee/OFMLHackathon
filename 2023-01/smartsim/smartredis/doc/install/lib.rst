
Clone the SmartRedis repository and optionally checkout a specific branch or tag:

.. code-block:: bash

    git clone https://github.com/CrayLabs/SmartRedis.git [--branch tag_name] smartredis

The release tarball can also be used instead of cloning the git repository, but
the preferred method is a repository clone.

The ```Makefile`` included in the top level of the SmartRedis repository has two
main targets: ``lib`` which will create a dynamic library for C, C++, and Python
clients and ``lib-with-fortran`` which will also additionally build a library
for Fortran applications. ``make help`` will list additional targets that are
used for SmartRedis development.

.. code-block:: bash

  cd SmartRedis
  make lib #or lib-with-fortran

The SmartRedis library will be installed in ``SmartRedis/install/lib/`` and the
SmartRedis header files (and optionally the Fortran ``.mod`` files) will be
installed in ``SmartRedis/install/include/``.  The library installation can be
used to easily include SmartRedis capabilities in C++, C, and Fortran
applications.

Linking instructions using compiler flags
-----------------------------------------

For applications which use pre-defined compiler flags for compilation, the
following flags should be included for the preprocessor

.. code-block:: text

    -I/path/to/smartredis/install/include

The linking flags will differ slightly whether the Fortran client library needs
to be included. If so, be sure that you ran ``make lib-with-fortran`` and
include the SmartRedis fortran library in the following flags

.. code-block:: text

    -L/path/to/smartredis/install/lib -lhiredis -lredis++ -lsmartredis [-lsmartredis-fortran]

.. note::

    Fortran applications need to link in both ``smartredis-fortran`` and
    ``smartredis`` libraries whereas C/C++ applications require only
    ``smartredis``


Linking instructions for CMake-based build systems
--------------------------------------------------

The CMake instructions below illustrate how to compile a C or C++ application
with SmartRedis. To build a Fortran client, uncomment out the lines after the
``Fortran-only`` comments

.. code-block:: text

    project(Example)

    cmake_minimum_required(VERSION 3.13)

    set(CMAKE_CXX_STANDARD 17)

    set(SMARTREDIS_INSTALL_PATH /path/to/smartredis/install)
    find_library(SMARTREDIS_LIBRARY smartredis
                 PATHS ${SMARTREDIS_INSTALL_PATH}/lib
                 NO_DEFAULT_PATH REQUIRED
    )

    # Fortran-only:
    #find_library(SMARTREDIS_FORTRAN_LIBRARY smartredis-fortran
    #             PATHS SMARTREDIS_INSTALL_PATH/lib
    #             NO_DEFAULT_PATH REQUIRED
    #)

    include_directories(SYSTEM
        /usr/local/include
        ${SMARTREDIS_INSTALL_PATH}/include
    )

    # Build executables

    add_executable(example
        example.cpp
    )
    target_link_libraries(example
        ${SMARTREDIS_LIBRARY}
        # Fortran-only:
        #${SMARTREDIS_FORTRAN_LIBRARY}
    )