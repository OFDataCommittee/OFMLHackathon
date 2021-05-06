project(CppClientExamples)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_STANDARD 17)

find_library(SR_LIB smartredis PATHS ../../../install/lib
             NO_DEFAULT_PATH REQUIRED)

include_directories(SYSTEM
    /usr/local/include
    ../../../install/include
)

# Build executables

add_executable(smartredis_put_get_3D
	smartredis_put_get_3D.cpp
)
target_link_libraries(smartredis_put_get_3D
	${SR_LIB}
)

add_executable(smartredis_dataset
	smartredis_dataset.cpp
)
target_link_libraries(smartredis_dataset
	${SR_LIB}
)

add_executable(smartredis_model
	smartredis_model.cpp
)
target_link_libraries(smartredis_model
	${SR_LIB}
)

add_executable(smartredis_mnist
	smartredis_mnist.cpp
)
target_link_libraries(smartredis_mnist
	${SR_LIB}
)
