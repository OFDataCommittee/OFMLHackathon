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

project(FortranClientExamples)

cmake_minimum_required(VERSION 3.13)

enable_language(Fortran)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

# Assume by default that users should link against the install directory in this repository
if(NOT DEFINED SMARTREDIS_INSTALL_PATH)
  set(SMARTREDIS_INSTALL_PATH "../../../install/")
endif()

# Specify all pre-processor and library dependencies
find_library(SMARTREDIS_LIBRARY smartredis PATHS ${SMARTREDIS_INSTALL_PATH}/lib NO_DEFAULT_PATH REQUIRED)
find_library(SMARTREDIS_FORTRAN_LIBRARY smartredis-fortran PATHS ${SMARTREDIS_INSTALL_PATH}/lib NO_DEFAULT_PATH REQUIRED)
find_library(HIREDIS hiredis PATHS ${SMARTREDIS_INSTALL_PATH}/lib NO_DEFAULT_PATH REQUIRED)
find_library(REDISPP redis++ PATHS ${SMARTREDIS_INSTALL_PATH}/lib NO_DEFAULT_PATH REQUIRED)
set(SMARTREDIS_LIBRARIES
	${SMARTREDIS_LIBRARY}
	${SMARTREDIS_FORTRAN_LIBRARY}
	${HIREDIS}
	${REDISPP}
)
include_directories(SYSTEM
    /usr/local/include
    ${SMARTREDIS_INSTALL_PATH}/include
)

# Define all the tests to be built
list(APPEND EXECUTABLES
     smartredis_dataset
     smartredis_put_get_3D
)

foreach(EXECUTABLE ${EXECUTABLES})

	add_executable(${EXECUTABLE}
		${EXECUTABLE}.F90
	)
	target_link_libraries(${EXECUTABLE}
		${SMARTREDIS_LIBRARIES}
	)
endforeach()