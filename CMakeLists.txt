project(SmartRedis)

cmake_minimum_required(VERSION 3.10)

option(BUILD_PYTHON "Build the python module" ON)

set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXE_LINKER_FLAGS "-lpthread")
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

find_library(REDISPP redis++ PATHS ${CMAKE_SOURCE_DIR}/install/lib NO_DEFAULT_PATH REQUIRED)
find_library(HIREDIS hiredis PATHS ${CMAKE_SOURCE_DIR}/install/lib NO_DEFAULT_PATH REQUIRED)

set(EXT_CLIENT_LIBRARIES ${REDISPP} ${HIREDIS})

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
    src/cpp/metadatafield.cpp
    src/cpp/stringfield.cpp
    src/fortran/fortran_c_interop.F90
    src/fortran/dataset.F90
    src/fortran/client.F90)

include_directories(SYSTEM
    include
    install/include
    src/fortran
)

# Build dynamic library
add_library(smartredis SHARED ${CLIENT_SRC})
set_target_properties(smartredis PROPERTIES SUFFIX ".so")
target_link_libraries(smartredis PUBLIC ${EXT_CLIENT_LIBRARIES})

# Install dynamic library and headers
install(TARGETS smartredis
	LIBRARY DESTINATION lib)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/"
        DESTINATION "include"
        FILES_MATCHING
        PATTERN "*.h" PATTERN "*.tcc" PATTERN "*.inc"
)

if(BUILD_PYTHON)
	message("-- Python client build enabled")
	add_subdirectory(${CMAKE_SOURCE_DIR}/third-party/pybind
                     ${CMAKE_SOURCE_DIR}/third-party/pybind/build)

    add_library(smartredis_static STATIC ${CLIENT_SRC})

	pybind11_add_module(smartredisPy
	                    src/python/src/pyclient.cpp
                        src/python/src/pydataset.cpp
                        ${CLIENT_SRC}
                        src/python/bindings/bind.cpp)

	target_link_libraries(smartredisPy PUBLIC ${EXT_CLIENT_LIBRARIES})
	install(TARGETS smartredisPy
	        LIBRARY DESTINATION lib)
else()
	message("-- Skipping Python client build")
endif()
