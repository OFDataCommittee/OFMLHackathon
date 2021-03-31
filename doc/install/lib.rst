
To build the static library for the C++, C, and Fortran clients in SmartRedis,
make sure to be in the top level directory of ``smartredis-0.1.0``.

.. code-block:: bash

  source setup_env.sh
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
