project(CppClientUnitTester)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS --coverage)


find_library(SR_LIB smartredis PATHS ../../../install/lib NO_DEFAULT_PATH REQUIRED)

include_directories(SYSTEM
    /usr/local/include
    ../../../include
	../../../install/include
)

set(SOURCES
	../../../src/cpp/client.cpp
	../../../src/cpp/dataset.cpp
	../../../src/cpp/command.cpp
	../../../src/cpp/commandlist.cpp
	../../../src/cpp/metadata.cpp
	../../../src/cpp/tensorbase.cpp
	../../../src/cpp/tensorpack.cpp
	../../../src/cpp/dbnode.cpp
	../../../src/cpp/commandreply.cpp
	../../../src/cpp/redisserver.cpp
	../../../src/cpp/rediscluster.cpp
	../../../src/cpp/redis.cpp
	../../../src/cpp/metadatafield.cpp
	../../../src/cpp/stringfield.cpp
)

set(UNIT_TESTS
	main.cpp
	test_client.cpp
	test_dataset.cpp
	test_tensorbase.cpp
	test_dbnode.cpp
	test_command.cpp
	test_commandlist.cpp
	test_tensor.cpp
	test_metadata.cpp
	test_stringfield.cpp
)

add_executable(test ${SOURCES} ${UNIT_TESTS})
target_link_libraries(test ${SR_LIB})
