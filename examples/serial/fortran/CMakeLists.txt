project(FortranClientExamples)

cmake_minimum_required(VERSION 3.10)

enable_language(Fortran)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

find_library(SR_LIB smartredis PATHS ../../../install/lib NO_DEFAULT_PATH REQUIRED)

include_directories(SYSTEM
    /usr/local/include
    ../../../install/include
)

set(ftn_client_src
    ../../../src/fortran/fortran_c_interop.F90
    ../../../src/fortran/dataset.F90
    ../../../src/fortran/client.F90
)

# build executables

add_executable(smartredis_put_get_3D
	smartredis_put_get_3D.F90
    ${ftn_client_src}
)
target_link_libraries(smartredis_put_get_3D
	${SR_LIB}
)

add_executable(smartredis_dataset
	smartredis_dataset.F90
    ${ftn_client_src}
)
target_link_libraries(smartredis_dataset
	${SR_LIB}
)
