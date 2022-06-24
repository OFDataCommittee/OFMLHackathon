
Clone the SmartRedis repository and checkout the most recent
release:

.. code-block:: bash

    git clone https://github.com/CrayLabs/SmartRedis.git --depth=1 --branch v0.3.1 smartredis

Note that the release tarball can also be used instead of cloning
the git repository, but the preferred method is a repository
clone.

To build the SmartRedis library for the C++, C, and Fortran,
make sure to be in the top level directory of ``smartredis-0.3.1``.

.. code-block:: bash

  make lib

The SmartRedis library will be installed in
``smartredis-0.3.1/install/lib/`` and the SmartRedis
header files will be installed in
``smartredis-0.3.1/install/include/``.
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
