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

# Build executables

add_executable(client_test_dataset
  client_test_dataset.cpp
)
target_link_libraries(client_test_dataset
  ${SR_LIB}
)

add_executable(client_test_dataset_no_tensors
  client_test_dataset_no_tensors.cpp
)
target_link_libraries(client_test_dataset_no_tensors
  ${SR_LIB}
)

add_executable(client_test_dataset_no_meta
  client_test_dataset_no_meta.cpp
)
target_link_libraries(client_test_dataset_no_meta
  ${SR_LIB}
)

add_executable(client_test_dataset_empty
  client_test_dataset_empty.cpp
)
target_link_libraries(client_test_dataset_empty
  ${SR_LIB}
)

add_executable(client_test_dataset_multiple_get_tensor
  client_test_dataset_multiple_get_tensor.cpp
)
target_link_libraries(client_test_dataset_multiple_get_tensor
  ${SR_LIB}
)

add_executable(client_test_delete_dataset
  client_test_delete_dataset.cpp
)
target_link_libraries(client_test_delete_dataset
  ${SR_LIB}
)

add_executable(client_test_rename_dataset
  client_test_rename_dataset.cpp
)
target_link_libraries(client_test_rename_dataset
  ${SR_LIB}
)

add_executable(client_test_copy_dataset
  client_test_copy_dataset.cpp
)
target_link_libraries(client_test_copy_dataset
  ${SR_LIB}
)

add_executable(client_test_dataset_copy_constructor
  client_test_dataset_copy_constructor.cpp
)
target_link_libraries(client_test_dataset_copy_constructor
  ${SR_LIB}
)

add_executable(client_test_dataset_move_constructor
  client_test_dataset_move_constructor.cpp
)
target_link_libraries(client_test_dataset_move_constructor
  ${SR_LIB}
)

add_executable(client_test_dataset_copy_assignment
  client_test_dataset_copy_assignment.cpp
)
target_link_libraries(client_test_dataset_copy_assignment
  ${SR_LIB}
)

add_executable(client_test_dataset_move_assignment
  client_test_dataset_move_assignment.cpp
)
target_link_libraries(client_test_dataset_move_assignment
  ${SR_LIB}
)

add_executable(client_test_put_get_3D
  client_test_put_get_3D.cpp
)
target_link_libraries(client_test_put_get_3D
  ${SR_LIB}
)

add_executable(client_test_put_get_3D_static_values
  client_test_put_get_3D_static_values.cpp
)
target_link_libraries(client_test_put_get_3D_static_values
  ${SR_LIB}
)

add_executable(client_test_put_get_contiguous_3D
  client_test_put_get_contiguous_3D.cpp
)
target_link_libraries(client_test_put_get_contiguous_3D
  ${SR_LIB}
)

add_executable(client_test_put_get_transpose_3D
  client_test_put_get_transpose_3D.cpp
)
target_link_libraries(client_test_put_get_transpose_3D
  ${SR_LIB}
)

add_executable(client_test_put_get_2D
  client_test_put_get_2D.cpp
)
target_link_libraries(client_test_put_get_2D
  ${SR_LIB}
)

add_executable(client_test_put_get_1D
  client_test_put_get_1D.cpp
)
target_link_libraries(client_test_put_get_1D
  ${SR_LIB}
)

add_executable(client_test_mnist
  client_test_mnist.cpp
)
target_link_libraries(client_test_mnist
  ${SR_LIB}
)

add_executable(client_test_mnist_dataset
  client_test_mnist_dataset.cpp
)
target_link_libraries(client_test_mnist_dataset
  ${SR_LIB}
)

add_executable(client_test_ensemble
  client_test_ensemble.cpp
)
target_link_libraries(client_test_ensemble
  ${SR_LIB}
)

add_executable(client_test_ensemble_dataset
  client_test_ensemble_dataset.cpp
)
target_link_libraries(client_test_ensemble_dataset
  ${SR_LIB}
)

