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

# Project definition for the SmartRedis-Examples-Serial-Fortran project
cmake_minimum_required(VERSION 3.13)
project(SmartRedis-Examples-Serial-Fortran)

# Enable language support for the examples
enable_language(Fortran)
enable_language(CXX)

# Configure the build
set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_C_STANDARD 99)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../../..")
include(smartredis_defs)

# Assume by default that users should link against the
# install directory in this repository
if(NOT DEFINED SMARTREDIS_INSTALL_PATH)
  set(SMARTREDIS_INSTALL_PATH "../../../install/")
endif()

# Locate dependencies
#  . Main SmartRedis Library (C/C++ based)
add_library(smartredis-main ${SMARTREDIS_LINK_MODE} IMPORTED)
find_library(SR_LIB ${SMARTREDIS_LIB}
    PATHS ${SMARTREDIS_INSTALL_PATH}/lib NO_DEFAULT_PATH
    REQUIRED
    ${SMARTREDIS_LINK_MODE}
)
set_target_properties(smartredis-main PROPERTIES
    IMPORTED_LOCATION ${SR_LIB}
)
#  . SmartRedis Fortran Library (Fortran based)
add_library(smartredis-fortran ${SMARTREDIS_LINK_MODE} IMPORTED)
find_library(SR_FTN_LIB ${SMARTREDIS_FORTRAN_LIB}
	PATHS ${SMARTREDIS_INSTALL_PATH}/lib NO_DEFAULT_PATH
	REQUIRED
    ${SMARTREDIS_LINK_MODE}
)
set_target_properties(smartredis-fortran PROPERTIES
    IMPORTED_LOCATION ${SR_FTN_LIB}
)

# Select libraries for build
if (STATIC_BUILD)
	# The CMake "preferred" approach only seems to work with the GNU
	# compiler. We will streamline this in the future
	if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
		# Mark that SmartRedis requires the C++ linker
		set_target_properties(smartredis-main PROPERTIES
			IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
		)
		set_target_properties(smartredis-fortran PROPERTIES
			IMPORTED_LINK_INTERFACE_LANGUAGES "FORTRAN"
		)
	else() # Tested with PGI, Intel
		# For other compilers, don't set languages so that CMake will use the Fortran linker (default)

		# Add the stdc++ linker flag
		set(CMAKE_EXE_LINKER_FLAGS "-lstdc++ ${CMAKE_EXE_LINKER_FLAGS}")
	endif()

	# Static builds have an extra dependency on the Pthreads library
	# The order of libraries here is crucial to get dependencies covered
	find_package(Threads REQUIRED)
	set(SMARTREDIS_LIBRARIES
		smartredis-fortran
		smartredis-main
		Threads::Threads
	)
else()
    # Shared builds only need the SmartRedis libraries
    set(SMARTREDIS_LIBRARIES
		smartredis-fortran
		smartredis-main
    )
endif()

# Define include directories for header files
include_directories(SYSTEM
    /usr/local/include
    ${SMARTREDIS_INSTALL_PATH}/include
)

# Define all the examples to be built
list(APPEND EXECUTABLES
     smartredis_dataset
     smartredis_put_get_3D
)

# Build the examples
foreach(EXECUTABLE ${EXECUTABLES})
	add_executable(${EXECUTABLE}_fortran_serial
		${EXECUTABLE}.F90
	)
    set_target_properties(${EXECUTABLE}_fortran_serial PROPERTIES
        OUTPUT_NAME ${EXECUTABLE}
    )
	target_link_libraries(${EXECUTABLE}_fortran_serial
		${SMARTREDIS_LIBRARIES}
	)
endforeach()
