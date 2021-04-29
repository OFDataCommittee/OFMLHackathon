project(CppClientTester)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_STANDARD 17)

find_library(SR_LIB smartredis PATHS ../../install/lib NO_DEFAULT_PATH REQUIRED)

include_directories(SYSTEM
    /usr/local/include
    ../../install/include
)

set(CLIENT_LIBRARIES ${SR_LIB})

# Build executables

add_executable(client_test_dataset
	client_test_dataset.cpp
)
target_link_libraries(client_test_dataset
	${CLIENT_LIBRARIES}
)

add_executable(client_test_delete_dataset
	client_test_delete_dataset.cpp
)
target_link_libraries(client_test_delete_dataset
	${CLIENT_LIBRARIES}
)

add_executable(client_test_rename_dataset
	client_test_rename_dataset.cpp
)
target_link_libraries(client_test_rename_dataset
	${CLIENT_LIBRARIES}
)

add_executable(client_test_copy_dataset
	client_test_copy_dataset.cpp
)
target_link_libraries(client_test_copy_dataset
	${CLIENT_LIBRARIES}
)

add_executable(client_test_dataset_copy_constructor
	client_test_dataset_copy_constructor.cpp
)
target_link_libraries(client_test_dataset_copy_constructor
	${CLIENT_LIBRARIES}
)

add_executable(client_test_dataset_move_constructor
	client_test_dataset_copy_constructor.cpp
)
target_link_libraries(client_test_dataset_move_constructor
	${CLIENT_LIBRARIES}
)

add_executable(client_test_dataset_copy_assignment
	client_test_dataset_copy_assignment.cpp
)
target_link_libraries(client_test_dataset_copy_assignment
	${CLIENT_LIBRARIES}
)

add_executable(client_test_dataset_move_assignment
	client_test_dataset_copy_assignment.cpp
)
target_link_libraries(client_test_dataset_move_assignment
	${CLIENT_LIBRARIES}
)

add_executable(client_test_put_get_3D
	client_test_put_get_3D.cpp
)
target_link_libraries(client_test_put_get_3D
	${CLIENT_LIBRARIES}
)

add_executable(client_test_put_get_3D_static_values
	client_test_put_get_3D_static_values.cpp
)
target_link_libraries(client_test_put_get_3D_static_values
	${CLIENT_LIBRARIES}
)

add_executable(client_test_put_get_contiguous_3D
	client_test_put_get_contiguous_3D.cpp
)
target_link_libraries(client_test_put_get_contiguous_3D
	${CLIENT_LIBRARIES}
)

add_executable(client_test_put_get_transpose_3D
	client_test_put_get_transpose_3D.cpp
)
target_link_libraries(client_test_put_get_transpose_3D
	${CLIENT_LIBRARIES}
)

add_executable(client_test_put_get_2D
	client_test_put_get_2D.cpp
)
target_link_libraries(client_test_put_get_2D
	${CLIENT_LIBRARIES}
)

add_executable(client_test_put_get_1D
	client_test_put_get_1D.cpp
)
target_link_libraries(client_test_put_get_1D
	${CLIENT_LIBRARIES}
)

add_executable(client_test_mnist
	client_test_mnist.cpp
	${CLIENT_SRC}
)
target_link_libraries(client_test_mnist
	${CLIENT_LIBRARIES}
)

add_executable(client_test_mnist_dataset
	client_test_mnist_dataset.cpp
	${CLIENT_SRC}
)
target_link_libraries(client_test_mnist_dataset
	${CLIENT_LIBRARIES}
)

add_executable(client_test_ensemble
	client_test_ensemble.cpp
)
target_link_libraries(client_test_ensemble
	${CLIENT_LIBRARIES}
)
