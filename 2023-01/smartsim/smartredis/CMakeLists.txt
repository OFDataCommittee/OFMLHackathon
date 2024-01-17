# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Enable setting version in the project statement
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

# Project definition for the SmartRedis project
cmake_minimum_required(VERSION 3.13)
project(SmartRedis VERSION "0.4.2")

# Configure options for the SmartRedis project
option(SR_PYTHON  "Build the python module" OFF)
option(SR_FORTRAN "Build the fortran client library" OFF)
option(SR_PEDANTIC "Build with pickiest compiler settings" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
set(CMAKE_CXX_VISIBILITY_PRESET default)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})
include(smartredis_defs)

# If we want to use Fortran, we have to tell CMake to use it
if (SR_FORTRAN)
    enable_language(Fortran)
endif()

# For now, we only support Pedantic on the main library build.
# If/when we fine-tune the examples and test cases, move this block
# to smartredis_defs.cmake
if (SR_PEDANTIC)
    if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND (CMAKE_C_COMPILER_ID STREQUAL "GNU"))
        add_compile_options(-Wall -Werror)
    else()
        message(WARNING "SR_PEDANTIC was specified, but the CMAKE compiler is not GCC")
    endif()
    if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wno-maybe-uninitialized")
    endif()
endif()

# Bring in third-party libaries needed for the SmartRedis library
find_library(REDISPP redis++
    PATHS ${CMAKE_SOURCE_DIR}/install/lib NO_DEFAULT_PATH
    REQUIRED STATIC
)
find_library(HIREDIS hiredis
    PATHS ${CMAKE_SOURCE_DIR}/install/lib NO_DEFAULT_PATH
    REQUIRED STATIC
)
find_package(Threads REQUIRED)
set(EXT_CLIENT_LIBRARIES ${REDISPP} ${HIREDIS})

# Define source code that goes into the SmartRedis library
set(CLIENT_SRC
    src/c/c_client.cpp
    src/c/c_configoptions.cpp
    src/c/c_dataset.cpp
    src/c/c_error.cpp
    src/c/c_logcontext.cpp
    src/c/c_logger.cpp
    src/cpp/address.cpp
    src/cpp/addressallcommand.cpp
    src/cpp/addressanycommand.cpp
    src/cpp/addressatcommand.cpp
    src/cpp/client.cpp
    src/cpp/clusterinfocommand.cpp
    src/cpp/command.cpp
    src/cpp/commandlist.cpp
    src/cpp/commandreply.cpp
    src/cpp/compoundcommand.cpp
    src/cpp/configoptions.cpp
    src/cpp/dataset.cpp
    src/cpp/dbinfocommand.cpp
    src/cpp/dbnode.cpp
    src/cpp/gettensorcommand.cpp
    src/cpp/keyedcommand.cpp
    src/cpp/logger.cpp
    src/cpp/metadata.cpp
    src/cpp/metadatafield.cpp
    src/cpp/multikeycommand.cpp
    src/cpp/nonkeyedcommand.cpp
    src/cpp/pipelinereply.cpp
    src/cpp/redis.cpp
    src/cpp/rediscluster.cpp
    src/cpp/redisserver.cpp
    src/cpp/singlekeycommand.cpp
    src/cpp/srobject.cpp
    src/cpp/stringfield.cpp
    src/cpp/tensorbase.cpp
    src/cpp/tensorpack.cpp
    src/cpp/threadpool.cpp
    src/cpp/utility.cpp
)

# Define include directories for header files
include_directories(SYSTEM
    include
    install/include
)

# Build the main SmartRedis library
if (STATIC_BUILD)
    set(TEMP_LIB_NAME sr${SR_STATIC_NAME_SUFFIX}-static)
    set(TEMP_LIB_FULLNAME lib${TEMP_LIB_NAME}.a)
    set(SR_LIB_INSTALL_PATH ${CMAKE_SOURCE_DIR}/include/lib)
    add_library(smartredis ${SMARTREDIS_LINK_MODE} ${CLIENT_SRC})
    set_target_properties(smartredis PROPERTIES
        SUFFIX ${SMARTREDIS_LINK_LIBRARY_SUFFIX}
    )
    set_target_properties(smartredis PROPERTIES
        OUTPUT_NAME ${TEMP_LIB_NAME}
    )
    target_link_libraries(smartredis PUBLIC ${EXT_CLIENT_LIBRARIES} PRIVATE Threads::Threads)

    # Merge SmartRedis static library with dependencies
    #  . Create a sacrificial dummy file so we have source to make a target against
    set(DUMMY_FILE ${CMAKE_BINARY_DIR}/merge-dummy.cpp)
    file(
        WRITE ${DUMMY_FILE}
        "// Life, the universe, and everything!\nstatic int sr_magic_number = 42;\n"
    )
    add_library(${SMARTREDIS_LIB} STATIC ${DUMMY_FILE})
    set(STATIC_LIB_COMPONENTS ${SR_LIB_INSTALL_PATH}/${TEMP_LIB_FULLNAME} ${EXT_CLIENT_LIBRARIES})

    #  . Archive the static libraries together into a thin library
    add_custom_command(
        TARGET smartredis
        POST_BUILD
        COMMAND rm -f lib${SMARTREDIS_LIB}.a && mkdir -p ${SR_LIB_INSTALL_PATH} && cp ${CMAKE_BINARY_DIR}/${TEMP_LIB_FULLNAME} ${SR_LIB_INSTALL_PATH} && ar crT lib${SMARTREDIS_LIB}.a ${STATIC_LIB_COMPONENTS}
        VERBATIM
        COMMENT "Bundling static libraries together as lib${SMARTREDIS_LIB}.a"
    )

    # Install static library
    install(TARGETS ${SMARTREDIS_LIB}
     	LIBRARY DESTINATION lib)
else () # Shared library build
    add_library(smartredis ${SMARTREDIS_LINK_MODE} ${CLIENT_SRC})
    set_target_properties(smartredis PROPERTIES
        SUFFIX ${SMARTREDIS_LINK_LIBRARY_SUFFIX}
    )
    set_target_properties(smartredis PROPERTIES
        OUTPUT_NAME ${SMARTREDIS_LIB}
    )
    target_link_libraries(smartredis PUBLIC ${EXT_CLIENT_LIBRARIES} PRIVATE Threads::Threads)

    # Install dynamic library
    install(TARGETS smartredis
     	LIBRARY DESTINATION lib)
endif()

# Install SmartRedis header files
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/"
        DESTINATION "include"
        FILES_MATCHING
        PATTERN "*.h" PATTERN "*.tcc" PATTERN "*.inc"
)

# Build the Fortran library
if (SR_FORTRAN)
    set(FORTRAN_SRC
        src/fortran/errors.F90
        src/fortran/client.F90
        src/fortran/configoptions.F90
        src/fortran/dataset.F90
        src/fortran/fortran_c_interop.F90
        src/fortran/logcontext.F90
        src/fortran/logger.F90
    )
    include_directories(src/fortran)
    # Note the following has to be before ANY add_library command)
    set(CMAKE_Fortran_MODULE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/include")
    # Fortran library
    add_library(smartredis-fortran ${SMARTREDIS_LINK_MODE} ${FORTRAN_SRC})
    set_target_properties(smartredis-fortran PROPERTIES
        SUFFIX ${SMARTREDIS_LINK_LIBRARY_SUFFIX}
    )
	set_target_properties(smartredis-fortran PROPERTIES
		OUTPUT_NAME ${SMARTREDIS_FORTRAN_LIB}
	)
    target_link_libraries(smartredis-fortran PUBLIC smartredis ${EXT_CLIENT_LIBRARIES})
    # Install dynamic library and headers
    install(TARGETS smartredis-fortran
    	LIBRARY DESTINATION lib)
endif()

# Build the Python library for SmartRedis
if(SR_PYTHON)
	message("-- Python client build enabled")
	add_subdirectory(${CMAKE_SOURCE_DIR}/third-party/pybind
                     ${CMAKE_SOURCE_DIR}/third-party/pybind/build)

	pybind11_add_module(smartredisPy
	                    src/python/src/pyclient.cpp
                        src/python/src/pyconfigoptions.cpp
                        src/python/src/pydataset.cpp
                        src/python/src/pylogcontext.cpp
                        src/python/src/pysrobject.cpp
                        ${CLIENT_SRC}
                        src/python/bindings/bind.cpp)

	target_link_libraries(smartredisPy PUBLIC ${EXT_CLIENT_LIBRARIES})
	install(TARGETS smartredisPy LIBRARY DESTINATION lib)
    install(TARGETS smartredisPy LIBRARY DESTINATION ../src/python/module/smartredis)
else()
	message("-- Skipping Python client build")
endif()
