project(CClientExamples)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_C_STANDARD 99)

find_library(SR_LIB smartredis PATHS ../../../install/lib
             NO_DEFAULT_PATH REQUIRED)

include_directories(SYSTEM
    /usr/local/include
    ../../../install/include
)

add_executable(example_put_unpack_1D
	example_put_unpack_1D.c
)

target_link_libraries(example_put_unpack_1D
	${SR_LIB}
)

add_executable(example_put_get_3D
        example_put_get_3D.c
)

target_link_libraries(example_put_get_3D
        ${SR_LIB}
)

