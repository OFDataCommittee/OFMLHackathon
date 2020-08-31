project(CppClientTester)

set(CMAKE_BUILD_TYPE RELEASE)

cmake_minimum_required(VERSION 3.10)

SET(CMAKE_CXX_STANDARD 17)

#TODO put in output warnings
# Determine where the hiredis library is based on if an environment variable
# is set, and if not, it can be found in the $PATH directory
if(DEFINED ENV{HIREDIS_INSTALL_PATH})
	   string(CONCAT HIREDIS_LIB_PATH $ENV{HIREDIS_INSTALL_PATH} "/lib")
	   find_library(HIREDIS_LIB hiredis PATHS ${HIREDIS_LIB_PATH} NO_DEFAULT_PATH)
	   string(CONCAT HIREDIS_INCLUDE_PATH $ENV{HIREDIS_INSTALL_PATH} "/include/")
else()
     	   find_library(HIREDIS_LIB hiredis PATH_SUFFIXES "/lib")
	   find_path(HIREDIS_INCLUDE_PATH hiredis PATH_SUFFIXES "/include")
endif()

# Determine where the redis++ library is based on if an environment variable
# is set, and if not, it can be found in the $PATH directory
if(DEFINED ENV{REDISPP_INSTALL_PATH})
	   string(CONCAT REDISPP_LIB_PATH $ENV{REDISPP_INSTALL_PATH} "/lib")
	   find_library(REDISPP_LIB redis++ PATHS ${REDISPP_LIB_PATH} )
	   string(CONCAT REDISPP_INCLUDE_PATH $ENV{REDISPP_INSTALL_PATH} "/include/")
else()
     	   find_library(REDISPP_LIB redis++ PATH_SUFFIXES "/lib")
	   find_path(REDISPP_INCLUDE_PATH redis++ PATH_SUFFIXES "/include")
endif()

find_package(MPI)

include_directories(SYSTEM
    ${MPI_INCLUDE_PATH}
    ../../include
    /usr/local/include
    ${HIREDIS_INCLUDE_PATH}
    ${REDISPP_INCLUDE_PATH}
)

set(CLIENT_LIBRARIES ${REDISPP_LIB} ${HIREDIS_LIB})
set(CLIENT_SRC ../../../src/cpp/client.cpp)


# Build executables
add_executable(client_test_binary
	client_test_binary.cpp
	${CLIENT_SRC}
)
target_link_libraries(client_test_binary
	MPI::MPI_CXX
	${CLIENT_LIBRARIES}
)

add_executable(client_test_binary_2D
	client_test_binary_2D.cpp
	${CLIENT_SRC}
)
target_link_libraries(client_test_binary_2D
	MPI::MPI_CXX
	${CLIENT_LIBRARIES}
)

add_executable(client_test_mnist
	client_test_mnist.cpp
	${CLIENT_SRC}
)
target_link_libraries(client_test_mnist
	MPI::MPI_CXX
	${CLIENT_LIBRARIES}
)
