
Docker Build
------------

SmartRedis is distributed with a Dockerfile to enable
the use of SmartRedis in application containers.
A sample Dockerfile is provided in
``./images/Dockerfile``, relative to the top-level
SmartRedis directory.  The SmartRedis Docker image
can be built by executing the following commands
in the top-level SmartRedis directory:

.. code-block:: bash

    cd ./images/Dockerfile
    bash ./build.sh

The ``build.sh`` script will execute ``docker build`` and
produce a ``smartredis`` image.

.. note::

    By default, the ``smartredis`` Docker image is built
    from copying the current contents of the SmartRedis
    top-level directory.  This means that any local changes
    that have been made will be incorporated into the
    Docker build.  To build from a separate repository
    and/or branch, a user can uncomment the ``WORKDIR`` and
    ``RUN`` commands provided in ``./images/Dockerfile`` and
    remove the ``COPY .  /usr/local/src/SmartRedis`` command.
    Comments have been left in the Dockefile to assist in making
    these changes.


Subsequent container applications can simply use the
Docker command ``FROM smartredis`` to build off of the
``smartredis`` image.  In the following sections,
additional details are provided for building and
running applications using the ``smartredis`` image.

Using the SmartRedis Docker Image
---------------------------------

Examples of building C++, C, Fortran, and Python applications
can be found in the the ``./tests/docker`` directory.  Below
are some notes and comments on building an application in each
language.

**C++ and C**

The ``smartredis`` Docker image has the SmartRedis dynamic
library installed in ``/usr/local/lib/``  and library
header files installed in ``/usr/local/include/smartredis/``.
An application can be built using the SmartRedis library
inside a container derived from the ``smartredis`` image
using these installed files.

Example CMake files that build C++ and C applications
using the ``smartredis` image can be found
at ``./tests/docker/cpp/CMakeLists.txt`` and
``./tests/docker/c/CMakeLists.txt``, respectively.
These CMake files are also shown below along with
example Docker files that invoke the CMake files
for a containerized application.

.. code-block:: text

    # C++ containerized CMake file

    cmake_minimum_required(VERSION 3.13)
    project(DockerTester)

    set(CMAKE_CXX_STANDARD 17)

    find_library(SR_LIB smartredis)

    include_directories(SYSTEM
        /usr/local/include/smartredis
    )

    # Build executables

    add_executable(docker_test
        docker_test.cpp
    )
    target_link_libraries(docker_test ${SR_LIB} pthread)


.. code-block:: docker

    # C++ Docker application Dockerfile

    FROM smartredis

    # Copy source and build files into the container
    COPY ./CMakeLists.txt /usr/local/src/DockerTest/
    COPY ./docker_test.cpp /usr/local/src/DockerTest/

    # Change working directory to the application folder
    WORKDIR /usr/local/src/DockerTest

    # Run CMake and make
    RUN mkdir build && cd build && cmake .. && make && cd ../

    # Default command for container execution
    CMD ["/usr/local/src/DockerTest/build/docker_test"]


.. code-block:: text

    # C containerized CMake file

    cmake_minimum_required(VERSION 3.13)
    project(DockerTester)

    set(CMAKE_CXX_STANDARD 17)

    find_library(SR_LIB smartredis)

    include_directories(SYSTEM
        /usr/local/include/smartredis
    )

    # Build executables

    add_executable(docker_test
        test_docker.c
    )
    target_link_libraries(docker_test ${SR_LIB} pthread)

.. code-block:: docker

    # C Docker application Dockerfile

    FROM smartredis

    # Copy source and build files into the container
    COPY ./CMakeLists.txt /usr/local/src/DockerTest/
    COPY ./test_docker.c /usr/local/src/DockerTest/

    # Change working directory to the application folder
    WORKDIR /usr/local/src/DockerTest

    # Run CMake and make
    RUN mkdir build && cd build && cmake .. && make && cd ../

    # Default command for container execution
    CMD ["/usr/local/src/DockerTest/build/docker_test"]

**Fortran**

The SmartRedis and SmartRedis-Fortran dynamic
library needed to compile a Fortran application
with SmartRedis are installed in ``/usr/local/lib/``
and the library header files are installed in
``/usr/local/include/smartredis/``.

An example CMake file that builds a Fortran application
using the ``smartredis`` images can be found
at ``./tests/docker/fortran/CMakeLists.txt``.
This CMake file is also shown below along with an
example Docker file that invokes the CMake files
for a containerized application.

.. code-block:: text

    # Fortran containerized CMake file

    cmake_minimum_required(VERSION 3.13)
    project(DockerTesterFortran)

    enable_language(Fortran)

    # Configure the build
    set(CMAKE_CXX_STANDARD 17)
    SET(CMAKE_C_STANDARD 99)
    set(CMAKE_BUILD_TYPE Debug)

    # Locate dependencies
    find_library(SR_LIB smartredis REQUIRED)
    find_library(SR_FTN_LIB smartredis-fortran REQUIRED)
    set(SMARTREDIS_LIBRARIES
        ${SR_LIB}
        ${SR_FTN_LIB}
    )

    # Define include directories for header files
    include_directories(SYSTEM
        /usr/local/include/smartredis
    )

    # Build the test
    add_executable(docker_test_fortran
        test_docker.F90
    )
    set_target_properties(docker_test_fortran PROPERTIES
        OUTPUT_NAME docker_test
    )
    target_link_libraries(docker_test_fortran ${SMARTREDIS_LIBRARIES} pthread)



.. code-block:: docker

    # Fortran Docker application Dockerfile

    FROM smartredis

    # Install Fortran compiler
    RUN apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
        gfortran && \
        rm -rf /var/lib/apt/lists/*

    # Copy source and build files into the container
    COPY ./CMakeLists.txt /usr/local/src/DockerTest/
    COPY ./test_docker.F90 /usr/local/src/DockerTest/

    # Change working directory to the application folder
    WORKDIR /usr/local/src/DockerTest

    # Run CMake and make
    RUN mkdir build && cd build && cmake .. && make && cd ../

    # Default command for container execution
    CMD ["/usr/local/src/DockerTest/build/docker_test"]

**Python**

The ``smartredis`` docker image includes the
SmartRedis Python module installed via pip into the
Python environment.  As a result, any containerized
Python script can import the SmartRedis module
without additional steps.

An example Dockerfile that containerizes a Python
script using SmartRedis is shown below and is
available at:
``./tests/docker/python/Dockerfile``.


.. code-block:: docker

    # Python Docker application Dockerfile

    FROM smartredis as builder

    # Copy application script
    COPY ./test_docker.py /usr/local/src/SmartRedis

    # Default command for container execution
    CMD ["python", "/usr/local/src/SmartRedis/test_docker.py"]
