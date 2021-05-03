project(CppClientExamples)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

find_package(MPI, REQUIRED)

find_library(SR_LIB smartredis PATHS ../../../install/lib NO_DEFAULT_PATH REQUIRED)

include_directories(SYSTEM
    /usr/local/include
    ${MPI_INCLUDE_PATH}
    ../../../install/include
)

set(ftn_client_src
    ../../../src/fortran/fortran_c_interop.F90
    ../../../src/fortran/dataset.F90
    ../../../src/fortran/client.F90
)

# Build executables

add_executable(smartredis_put_get_3D
	smartredis_put_get_3D.cpp
    ${ftn_client_src}
)
target_link_libraries(smartredis_put_get_3D
	MPI::MPI_CXX
	${SR_LIB}
)

add_executable(smartredis_mnist
	smartredis_mnist.cpp
    ${ftn_client_src}
)
target_link_libraries(smartredis_mnist
	MPI::MPI_CXX
	${SR_LIB}
)
