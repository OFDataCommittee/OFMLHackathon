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

# Set defaults if variables are undefined
if(NOT DEFINED SR_BUILD)
    set(SR_BUILD "Release")
endif()
if(NOT DEFINED SR_LINK)
    set(SR_LINK "Shared")
endif()

# Configure the CMake build based on the SR_BUILD selection
string(TOLOWER "${SR_BUILD}" srbuild_lowercase)
if(srbuild_lowercase STREQUAL "release")
    set(CMAKE_BUILD_TYPE RELEASE)
    set(SRLIB_NAME_SUFFIX "")
    set(SR_STATIC_NAME_SUFFIX "-r")
elseif(srbuild_lowercase STREQUAL "debug")
    set(CMAKE_BUILD_TYPE DEBUG)
    set(SRLIB_NAME_SUFFIX "-debug")
    set(SR_STATIC_NAME_SUFFIX "-d")
elseif(srbuild_lowercase STREQUAL "coverage")
    set(CMAKE_BUILD_TYPE DEBUG)
    set(SRLIB_NAME_SUFFIX "-coverage")
    set(SR_STATIC_NAME_SUFFIX "-c")
    if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND (CMAKE_C_COMPILER_ID STREQUAL "GNU"))
        add_compile_options(--coverage)
        add_link_options(--coverage)
    else()
        message(WARNING "A coverage build was specified, but the CMAKE compiler is not GCC")
    endif()
else()
    message(FATAL_ERROR "Unrecognized build type (${SR_BUILD}) specified in SR_BUILD")
endif()

# Configure CMake linkage on the SR_LINK selection
string(TOLOWER "${SR_LINK}" srlink_lowercase)
if(srlink_lowercase STREQUAL "static")
    set(SMARTREDIS_LINK_MODE STATIC)
    set(SMARTREDIS_LINK_LIBRARY_SUFFIX .a)
    set(SMARTREDIS_LINK_SUFFIX "-static")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(STATIC_BUILD TRUE)
    if(APPLE)
        message(FATAL_ERROR "Static builds are not supported on MacOS")
    endif(APPLE)
elseif(srlink_lowercase STREQUAL "shared")
    set(SMARTREDIS_LINK_MODE SHARED)
    set(SMARTREDIS_LINK_LIBRARY_SUFFIX .so)
    set(SMARTREDIS_LINK_SUFFIX "")
    set(STATIC_BUILD FALSE)
else()
    message(FATAL_ERROR "Unrecognized link type (${SR_LINK}) specified in SR_LINK")
endif()

# Identify the SmartRedis library names based on the build and link
set(SMARTREDIS_LIB smartredis${SRLIB_NAME_SUFFIX}${SMARTREDIS_LINK_SUFFIX})
set(SMARTREDIS_LIB_FULLNAME
    libsmartredis${SRLIB_NAME_SUFFIX}${SMARTREDIS_LINK_SUFFIX}${SMARTREDIS_LINK_LIBRARY_SUFFIX}
)
set(SMARTREDIS_FORTRAN_LIB smartredis-fortran${SRLIB_NAME_SUFFIX}${SMARTREDIS_LINK_SUFFIX})
set(SMARTREDIS_FORTRAN_LIB_FULLNAME
    libsmartredis-fortran${SRLIB_NAME_SUFFIX}${SMARTREDIS_LINK_SUFFIX}${SMARTREDIS_LINK_LIBRARY_SUFFIX}
)
