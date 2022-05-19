# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

project(FortranClientTester)

cmake_minimum_required(VERSION 3.13)

enable_language(Fortran)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

set(ftn_client_src
    ../../../src/fortran/fortran_c_interop.F90
    ../../../src/fortran/dataset.F90
    ../../../src/fortran/client.F90
)

find_library(SR_LIB smartredis PATHS ../../install/lib NO_DEFAULT_PATH REQUIRED)

include_directories(SYSTEM
    /usr/local/include
    ../../install/include
)

add_executable(client_test_put_get_1D
	client_test_put_get_1D.F90
	${ftn_client_src}
	test_utils.F90
)
target_link_libraries(client_test_put_get_1D
	${SR_LIB}
)

add_executable(client_test_put_get_2D
	client_test_put_get_2D.F90
	${ftn_client_src}
        test_utils.F90
)
target_link_libraries(client_test_put_get_2D
	${SR_LIB}
)

add_executable(client_test_put_get_3D
	client_test_put_get_3D.F90
	${ftn_client_src}
        test_utils.F90
)
target_link_libraries(client_test_put_get_3D
	${SR_LIB}
)

add_executable(client_test_mnist
	client_test_mnist.F90
	${ftn_client_src}
        test_utils.F90
)
target_link_libraries(client_test_mnist
	${SR_LIB}
)

add_executable(client_test_mnist_multigpu
	client_test_mnist_multigpu.F90
	${ftn_client_src}
        test_utils.F90
)
target_link_libraries(client_test_mnist_multigpu
	${SR_LIB}
)

add_executable(client_test_dataset
	client_test_dataset.F90
	${ftn_client_src}
        test_utils.F90
)
target_link_libraries(client_test_dataset
	${SR_LIB}
)

add_executable(client_test_put_get_unpack_dataset
	client_test_put_get_unpack_dataset.F90
	${ftn_client_src}
        test_utils.F90
)
target_link_libraries(client_test_put_get_unpack_dataset
	${SR_LIB}
)

add_executable(client_test_misc_tensor
	client_test_misc_tensor.F90
	${ftn_client_src}
        test_utils.F90
)
target_link_libraries(client_test_misc_tensor
	${SR_LIB}
)

add_executable(client_test_ensemble
	client_test_ensemble.F90
	${ftn_client_src}
        test_utils.F90
)
target_link_libraries(client_test_ensemble
	${SR_LIB}
)

add_executable(client_test_initialized
        client_test_initialized.F90
        ${ftn_client_src}
        test_utils.F90
)
target_link_libraries(client_test_initialized
	${SR_LIB}
)
