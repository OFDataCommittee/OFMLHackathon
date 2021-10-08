project(CClientTester)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_C_STANDARD 99)

find_library(SR_LIB smartredis PATHS ../../install/lib NO_DEFAULT_PATH REQUIRED)

include_directories(SYSTEM
    /usr/local/include
    ../../install/include
)

add_executable(client_test_dataset_exists
	client_test_dataset_exists.c
)

target_link_libraries(client_test_dataset_exists
	${SR_LIB}
)

add_executable(client_test_put_unpack_1D
	client_test_put_unpack_1D.c
)

target_link_libraries(client_test_put_unpack_1D
	${SR_LIB}
)

add_executable(client_test_put_get_1D
        client_test_put_get_1D.c
)

target_link_libraries(client_test_put_get_1D
        ${SR_LIB}
)

add_executable(client_test_put_get_2D
        client_test_put_get_2D.c
)

target_link_libraries(client_test_put_get_2D
        ${SR_LIB}
)

add_executable(client_test_put_get_3D
        client_test_put_get_3D.c
)

target_link_libraries(client_test_put_get_3D
        ${SR_LIB}
)

