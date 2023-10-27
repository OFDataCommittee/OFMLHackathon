
Clone the SmartRedis repository and optionally checkout a specific branch or tag:

.. code-block:: bash

    git clone https://github.com/CrayLabs/SmartRedis.git [--branch tag_name] smartredis

The release tarball can also be used instead of cloning the git repository, but
the preferred method is a repository clone.

The ```Makefile`` included in the top level of the SmartRedis repository has two
main targets: ``lib`` which will create a dynamic library for C, C++, and
(optionally) Fortran and Python clients; and ``lib-with-fortran`` which will also
unconditionally build a library for Fortran applications. ``make help`` will list
additional targets that are used for SmartRedis development.

.. code-block:: bash

  cd SmartRedis
  make lib #or lib-with-fortran

The SmartRedis library will be installed in ``SmartRedis/install/lib/`` and the
SmartRedis header files (and optionally the Fortran ``.mod`` files) will be
installed in ``SmartRedis/install/include/``.  The library installation can be
used to easily include SmartRedis capabilities in C++, C, and Fortran
applications.

Customizing the library build
-----------------------------

By default, the SmartRedis library is built as a shared library. For some
applications, however, it is preferable to link to a statically compiled
library. This can be done easily with the command:

.. code-block:: bash

    cd SmartRedis
    # Static build
    make lib SR_LINK=Static
    # Shared build
    make lib SR_LINK=Shared #or skip the SR_LINK variable as this is the default

Linked statically, the SmartRedis library will have a ``.a`` file extension.  When
linked dynamically, the SmartRedis library will have a ``.so`` file extension.

It is also possible to adjust compilation settings for the SmartRedis library.
By default, the library compiles in an optimized build (``Release``), but debug builds
with full symbols (``Debug``) can be created as can debug builds with extensions enabled
for code coverage metrics (``Coverage``; this build type is only available with GNU
compilers). Similar to configuring a link type, selecting the build mode can be done
via a variable supplied to make:

.. code-block:: bash

    cd SmartRedis
    # Release build
    make lib SR_BUILD=Release #or skip the SR_BUILD variable as this is the default
    # Debug build
    make lib SR_BUILD=Debug
    # Code coverage build
    make lib SR_BUILD=Coverage

The name of the library produced for a Debug mode build is ``smartredis-debug``.
The name of the library produced for a Coverage mode build is ``smartredis-coverage``.
The name of the library  produced for a Release mode build is ``smartredis``.
In each case, the file extension is dependent on the link type, ``.so`` or ``.a``.
All libraries will be located in the ``install/lib`` folder.

Finally, it is possible to build SmartRedis to include Python and/or Fortran support
(both are omitted by default):

.. code-block:: bash

    cd SmartRedis
    # Build support for Python
    make lib SR_PYTHON=ON
    # Build support for Fortran
    make lib SR_FORTRAN=ON # equivalent to make lib-with-fortran
    # Build support for Python and Fortran
    make lib SR_PYTHON=ON SR_FORTRAN=ON # or make lib-with-fortran SR_PYTHON=ON

The build mode, link type, and Fortran/Python support settings are fully orthogonal;
any combination of them is supported. For example, a statically linked debug build
with Python support may be achieved via the following command:

.. code-block:: bash

    cd SmartRedis
    make lib SR_LINK=Static SR_BUILD=Debug SR_PYTHON=ON

The SR_LINK, SR_BUILD, SR_PYTHON, and SR_FORTRAN variables are fully supported for all
test and build targets in the Makefile.

Fortran support is built in a secondary library.
The name of the Fortran library produced for a Debug mode build is ``smartredis-fortran-debug``.
The name of the library produced for a Coverage mode build is ``smartredis-fortran-coverage``.
The name of the library  produced for a Release mode build is ``smartredis-fortran``.
As with the main libray, the file extension is dependent on the link type, ``.so`` or ``.a``.
All libraries will be located in the ``install/lib`` folder.


Additional make variables are described in the ``help`` make target:

.. code-block:: bash

    cd SmartRedis
    make help

Linking instructions using compiler flags
-----------------------------------------

For applications which use pre-defined compiler flags for compilation, the
following flags should be included for the preprocessor

.. code-block:: text

    -I/path/to/smartredis/install/include

The linking flags will differ slightly whether the Fortran client library needs
to be included. If so, be sure that you ran ``make lib-with-fortran`` (or ``make
lib SR_FORTRAN=ON``) and include the SmartRedis fortran library via the following flags:

.. code-block:: text

    -L/path/to/smartredis/install/lib -lsmartredis [-lsmartredis-fortran]

.. note::

    Fortran applications need to link in both ``smartredis-fortran`` and
    ``smartredis`` libraries whereas C/C++ applications require only
    ``smartredis``. For debug or coverage builds, use the appropriate alternate
    libraries as described previously.


Linking instructions for CMake-based build systems
--------------------------------------------------

The CMake instructions below illustrate how to compile a C or C++ application
with SmartRedis. To build a Fortran client, uncomment out the lines after the
``Fortran-only`` comments

.. code-block:: text

    cmake_minimum_required(VERSION 3.13)
    project(Example)

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