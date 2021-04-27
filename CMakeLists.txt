project(SmartRedis)

set(CMAKE_BUILD_TYPE Release)

cmake_minimum_required(VERSION 3.10)

SET(CMAKE_CXX_STANDARD 17)
enable_language(Fortran)
SET(CMAKE_EXE_LINKER_FLAGS "-lpthread")

option(BUILD_PYTHON "Build the python module" ON)

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

# Determine where the protobuf library is based on if an environment variable
# is set, and if not, it can be found in the $PATH directory
if(DEFINED ENV{PROTOBUF_INSTALL_PATH})
	   string(CONCAT PROTOBUF_LIB_PATH $ENV{PROTOBUF_INSTALL_PATH} "/lib")
	   find_library(PROTOBUF_LIB protobuf PATHS ${PROTOBUF_LIB_PATH} NO_DEFAULT_PATH)
	   string(CONCAT PROTOBUF_INCLUDE_PATH $ENV{PROTOBUF_INSTALL_PATH} "/include/")
else()
     	   find_library(PROTOBUF_LIB protobuf PATH_SUFFIXES "/lib")
	   find_path(PROTOBUF_INCLUDE_PATH protobuf PATH_SUFFIXES "/include")
endif()

include_directories(SYSTEM
    include
    utils/protobuf
    src/fortran
    /usr/local/include
    ${HIREDIS_INCLUDE_PATH}
    ${REDISPP_INCLUDE_PATH}
    ${PROTOBUF_INCLUDE_PATH}
)

set(CLIENT_LIBRARIES ${REDISPP_LIB} ${HIREDIS_LIB} ${PROTOBUF_LIB})

add_library( smartredis STATIC
	 src/c/c_client.cpp
	 src/c/c_dataset.cpp
	 src/cpp/client.cpp
	 src/cpp/dataset.cpp
	 src/cpp/command.cpp
	 src/cpp/commandlist.cpp
	 src/cpp/metadata.cpp
	 src/cpp/tensorbase.cpp
	 src/cpp/tensorpack.cpp
	 src/cpp/dbnode.cpp
	 src/cpp/commandreply.cpp
     src/cpp/redisserver.cpp
     src/cpp/rediscluster.cpp
     src/cpp/redis.cpp
	 utils/protobuf/smartredis.pb.cc
   src/fortran/fortran_c_interop.F90
   src/fortran/dataset.F90
   src/fortran/client.F90)


if(BUILD_PYTHON)
	message("-- Python client build enabled")
	add_subdirectory($ENV{PYBIND_INSTALL_PATH} $ENV{PYBIND_INSTALL_PATH}/build)

	set(CLIENT_SRC
		src/c/c_client.cpp
		src/c/c_dataset.cpp
		src/cpp/client.cpp
		src/cpp/dataset.cpp
		src/cpp/command.cpp
		src/cpp/commandlist.cpp
		src/cpp/metadata.cpp
		src/cpp/tensorbase.cpp
		src/cpp/tensorpack.cpp
		src/cpp/dbnode.cpp
		src/cpp/commandreply.cpp
		src/cpp/redisserver.cpp
		src/cpp/rediscluster.cpp
		src/cpp/redis.cpp
		utils/protobuf/smartredis.pb.cc
		src/python/src/pyclient.cpp
		src/python/src/pydataset.cpp)

	pybind11_add_module(smartredisPy ${CLIENT_SRC} ${CLIENT_LIBRARIES} src/python/bindings/bind.cpp)
	target_link_libraries(smartredisPy PUBLIC ${CLIENT_LIBRARIES})
else()
	message("-- Skipping Python client build")
endif()