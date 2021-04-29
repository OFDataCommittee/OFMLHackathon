project(SmartRedis)

cmake_minimum_required(VERSION 3.10)

option(BUILD_PYTHON "Build the python module" ON)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXE_LINKER_FLAGS "-lpthread")
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

find_library(REDISPP redis++ PATHS ./install/lib NO_DEFAULT_PATH REQUIRED)
find_library(HIREDIS hiredis PATHS ./install/lib NO_DEFAULT_PATH REQUIRED)
find_library(PROTOBUF protobuf PATHS ./install/lib NO_DEFAULT_PATH REQUIRED)

set(EXT_CLIENT_LIBRARIES ${REDISPP} ${HIREDIS} ${PROTOBUF})

set(CLIENT_SRC
    src/c/c_client.cpp
    src/c/c_dataset.cpp
    src/cpp/client.cpp
    src/cpp/dataset.cpp
    src/cpp/command.cpp
    src/cpp/commandlist.cpp
    src/cpp/metadata.cpp
    src/cpp/tensorbase.cpp
    src/cpp/tensorpack.cpp
    src/cpp/dbnode.cpp
    src/cpp/commandreply.cpp
    src/cpp/redisserver.cpp
    src/cpp/rediscluster.cpp
    src/cpp/redis.cpp
    utils/protobuf/smartredis.pb.cc
    src/fortran/fortran_c_interop.F90
    src/fortran/dataset.F90
    src/fortran/client.F90)

include_directories(SYSTEM
    install/include/
    include
    utils/protobuf
    src/fortran
)

# Build dynamic library
add_library(smartredis SHARED ${CLIENT_SRC})
target_link_libraries(smartredis ${link_flag} ${EXT_CLIENT_LIBRARIES})

# Install dynamic library and headers
install(TARGETS smartredis
	LIBRARY DESTINATION lib)
	
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/" # source directory
        DESTINATION "include" # target directory
        FILES_MATCHING # install only matched files
        PATTERN "*.h" PATTERN "*.tcc" # select header files
)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/utils/protobuf/" # source directory
        DESTINATION "include" # target directory
        FILES_MATCHING # install only matched files
        PATTERN "*.h" PATTERN "*.tcc" # select header files
)

if(BUILD_PYTHON)
	message("-- Python client build enabled")
	add_subdirectory($ENV{PYBIND_INSTALL_PATH} $ENV{PYBIND_INSTALL_PATH}/build)
	pybind11_add_module(smartredisPy ${CLIENT_SRC} ${EXT_CLIENT_LIBRARIES} src/python/bindings/bind.cpp)
	target_link_libraries(smartredisPy PUBLIC ${EXT_CLIENT_LIBRARIES})
else()
	message("-- Skipping Python client build")
endif()
