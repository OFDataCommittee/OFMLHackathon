project(FortranClientTester)

cmake_minimum_required(VERSION 3.10)

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
    ../../include
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
